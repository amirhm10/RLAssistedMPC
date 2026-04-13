from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.optimize as spo

from utils.state_features import get_rl_state_dim


def _as_1d_float(x, expected_len: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if expected_len is not None and arr.size != int(expected_len):
        raise ValueError(f"Expected vector of length {expected_len}, got {arr.size}.")
    return arr


def _relative_fro(candidate: np.ndarray, nominal: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=float)
    nominal = np.asarray(nominal, dtype=float)
    denom = float(np.linalg.norm(nominal, ord="fro"))
    if denom <= 0.0:
        return 0.0
    return float(np.linalg.norm(candidate - nominal, ord="fro") / denom)


def _validate_physical_model_shapes(A0_phys: np.ndarray, B0_phys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A0_phys = np.asarray(A0_phys, dtype=float)
    B0_phys = np.asarray(B0_phys, dtype=float)
    if A0_phys.ndim != 2 or A0_phys.shape[0] != A0_phys.shape[1]:
        raise ValueError("A0_phys must be a square 2D array.")
    if B0_phys.ndim != 2 or B0_phys.shape[0] != A0_phys.shape[0]:
        raise ValueError("B0_phys must be a 2D array with the same number of rows as A0_phys.")
    return A0_phys, B0_phys


def _resolve_contiguous_block_groups(n_phys: int, block_group_count: int) -> list[np.ndarray]:
    n_phys = int(n_phys)
    block_group_count = int(block_group_count)
    if n_phys <= 0:
        raise ValueError("n_phys must be positive.")
    if block_group_count <= 0:
        raise ValueError("block_group_count must be positive.")
    block_group_count = min(block_group_count, n_phys)
    return [np.asarray(chunk, dtype=int) for chunk in np.array_split(np.arange(n_phys, dtype=int), block_group_count)]


def _normalize_block_groups(
    n_phys: int,
    block_groups: Sequence[Sequence[int]] | None,
    block_group_count: int,
) -> list[np.ndarray]:
    if block_groups is None:
        groups = _resolve_contiguous_block_groups(n_phys=n_phys, block_group_count=block_group_count)
    else:
        groups = []
        seen: set[int] = set()
        for idx, group in enumerate(block_groups):
            arr = np.asarray(group, dtype=int).reshape(-1)
            if arr.size == 0:
                raise ValueError(f"block_groups[{idx}] is empty.")
            if np.any(arr < 0) or np.any(arr >= int(n_phys)):
                raise ValueError("block_groups contains indices outside the physical-state range.")
            unique = np.unique(arr)
            if unique.size != arr.size:
                raise ValueError("block_groups entries must not repeat indices within a block.")
            overlap = set(int(v) for v in unique).intersection(seen)
            if overlap:
                raise ValueError("block_groups must be disjoint.")
            seen.update(int(v) for v in unique)
            groups.append(unique)
        if len(seen) != int(n_phys):
            raise ValueError("block_groups must cover every physical state exactly once.")
    return groups


def _make_basis_dict(
    *,
    basis_name: str,
    basis_family: str,
    A_basis: list[np.ndarray],
    B_basis: list[np.ndarray],
    theta_labels: list[str],
    block_groups: list[np.ndarray] | None = None,
) -> dict:
    return {
        "basis_name": basis_name,
        "basis_family": basis_family,
        "A_basis": [np.asarray(E, dtype=float) for E in A_basis],
        "B_basis": [np.asarray(F, dtype=float) for F in B_basis],
        "theta_labels": list(theta_labels),
        "n_A": len(A_basis),
        "n_B": len(B_basis),
        "theta_dim": len(theta_labels),
        "block_groups": None if block_groups is None else [np.asarray(group, dtype=int).copy() for group in block_groups],
    }


@dataclass
class RollingIDBatch:
    x_t: np.ndarray
    u_t: np.ndarray
    x_tp1: np.ndarray


class RollingIDBuffer:
    def __init__(self, maxlen: int, state_dim: int, input_dim: int):
        self.maxlen = int(maxlen)
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        if self.maxlen <= 0:
            raise ValueError("maxlen must be positive.")
        self._buffer: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=self.maxlen)

    def push(self, x_t, u_t, x_tp1) -> None:
        x_t = _as_1d_float(x_t, self.state_dim)
        u_t = _as_1d_float(u_t, self.input_dim)
        x_tp1 = _as_1d_float(x_tp1, self.state_dim)
        self._buffer.append((x_t.copy(), u_t.copy(), x_tp1.copy()))

    def __len__(self) -> int:
        return len(self._buffer)

    def get_recent(self, window: int | None = None) -> RollingIDBatch:
        if len(self._buffer) == 0:
            raise ValueError("Cannot build an identification batch from an empty buffer.")
        if window is None:
            items = list(self._buffer)
        else:
            window = int(window)
            if window <= 0:
                raise ValueError("window must be positive.")
            items = list(self._buffer)[-window:]
        x_t = np.stack([item[0] for item in items], axis=0)
        u_t = np.stack([item[1] for item in items], axis=0)
        x_tp1 = np.stack([item[2] for item in items], axis=0)
        return RollingIDBatch(x_t=x_t, u_t=u_t, x_tp1=x_tp1)


def build_polymer_scalar_legacy_basis(A0_phys: np.ndarray, B0_phys: np.ndarray) -> dict:
    """
    Legacy phase-1 identification basis:
    one global physical-A correction and one correction per B input column.
    """
    A0_phys, B0_phys = _validate_physical_model_shapes(A0_phys, B0_phys)
    A_basis = [A0_phys.copy()]
    B_basis = []
    labels = ["theta_A_global"]
    for col in range(B0_phys.shape[1]):
        basis_col = np.zeros_like(B0_phys)
        basis_col[:, col] = B0_phys[:, col]
        B_basis.append(basis_col)
        labels.append(f"theta_B_{col + 1}")
    return _make_basis_dict(
        basis_name="polymer_phase1_scalar_matrix_basis",
        basis_family="scalar_legacy",
        A_basis=A_basis,
        B_basis=B_basis,
        theta_labels=labels,
    )


def build_polymer_rowcol_basis(A0_phys: np.ndarray, B0_phys: np.ndarray) -> dict:
    """
    V2 row/column basis:
    - one A row-scaling basis per physical state
    - one B input-column basis per input
    """
    A0_phys, B0_phys = _validate_physical_model_shapes(A0_phys, B0_phys)
    n_phys = int(A0_phys.shape[0])
    A_basis = []
    labels = []
    for row in range(n_phys):
        projector = np.zeros((n_phys, n_phys), dtype=float)
        projector[row, row] = 1.0
        A_basis.append(projector @ A0_phys)
        labels.append(f"theta_A_row_{row + 1}")
    B_basis = []
    for col in range(B0_phys.shape[1]):
        basis_col = np.zeros_like(B0_phys)
        basis_col[:, col] = B0_phys[:, col]
        B_basis.append(basis_col)
        labels.append(f"theta_B_{col + 1}")
    return _make_basis_dict(
        basis_name="polymer_phase1_rowcol_basis",
        basis_family="rowcol",
        A_basis=A_basis,
        B_basis=B_basis,
        theta_labels=labels,
    )


def build_polymer_block_basis(
    A0_phys: np.ndarray,
    B0_phys: np.ndarray,
    *,
    block_groups: Sequence[Sequence[int]] | None = None,
    block_group_count: int = 3,
) -> dict:
    """
    V2 polymer block basis:
    - one A row-block basis per configured physical-state block
    - one B input-column basis per input
    """
    A0_phys, B0_phys = _validate_physical_model_shapes(A0_phys, B0_phys)
    resolved_groups = _normalize_block_groups(
        n_phys=A0_phys.shape[0],
        block_groups=block_groups,
        block_group_count=block_group_count,
    )
    A_basis = []
    labels = []
    for block_idx, group in enumerate(resolved_groups):
        projector = np.zeros((A0_phys.shape[0], A0_phys.shape[0]), dtype=float)
        projector[group, group] = 1.0
        A_basis.append(projector @ A0_phys)
        labels.append(f"theta_A_block_{block_idx + 1}")
    B_basis = []
    for col in range(B0_phys.shape[1]):
        basis_col = np.zeros_like(B0_phys)
        basis_col[:, col] = B0_phys[:, col]
        B_basis.append(basis_col)
        labels.append(f"theta_B_{col + 1}")
    return _make_basis_dict(
        basis_name="polymer_phase1_block_polymer_basis",
        basis_family="block_polymer",
        A_basis=A_basis,
        B_basis=B_basis,
        theta_labels=labels,
        block_groups=resolved_groups,
    )


def build_polymer_identification_basis(
    A0_phys: np.ndarray,
    B0_phys: np.ndarray,
    *,
    basis_family: str = "scalar_legacy",
    block_groups: Sequence[Sequence[int]] | None = None,
    block_group_count: int = 3,
) -> dict:
    basis_family = str(basis_family).lower()
    if basis_family == "scalar_legacy":
        return build_polymer_scalar_legacy_basis(A0_phys=A0_phys, B0_phys=B0_phys)
    if basis_family == "rowcol":
        return build_polymer_rowcol_basis(A0_phys=A0_phys, B0_phys=B0_phys)
    if basis_family == "block_polymer":
        return build_polymer_block_basis(
            A0_phys=A0_phys,
            B0_phys=B0_phys,
            block_groups=block_groups,
            block_group_count=block_group_count,
        )
    raise ValueError("basis_family must be 'scalar_legacy', 'rowcol', or 'block_polymer'.")


def build_polymer_phase1_basis(A0_phys: np.ndarray, B0_phys: np.ndarray) -> dict:
    return build_polymer_scalar_legacy_basis(A0_phys=A0_phys, B0_phys=B0_phys)


def assemble_batch_regression(batch: RollingIDBatch, A0_phys: np.ndarray, B0_phys: np.ndarray, basis: dict) -> tuple[np.ndarray, np.ndarray]:
    A0_phys = np.asarray(A0_phys, dtype=float)
    B0_phys = np.asarray(B0_phys, dtype=float)
    A_basis = [np.asarray(E, dtype=float) for E in basis["A_basis"]]
    B_basis = [np.asarray(F, dtype=float) for F in basis["B_basis"]]

    phi_blocks = []
    residual_blocks = []
    for x_t, u_t, x_tp1 in zip(batch.x_t, batch.u_t, batch.x_tp1):
        residual = x_tp1 - (A0_phys @ x_t) - (B0_phys @ u_t)
        feature_cols = [E @ x_t for E in A_basis] + [F @ u_t for F in B_basis]
        phi_t = np.column_stack(feature_cols)
        phi_blocks.append(phi_t)
        residual_blocks.append(residual)

    Phi = np.vstack(phi_blocks)
    residual_vec = np.concatenate(residual_blocks, axis=0)
    return Phi.astype(float), residual_vec.astype(float)


def solve_batch_ridge(
    Phi: np.ndarray,
    residual_vec: np.ndarray,
    theta_prev: np.ndarray,
    lambda_prev: float,
    lambda_0: float,
    theta_low: np.ndarray,
    theta_high: np.ndarray,
) -> dict:
    Phi = np.asarray(Phi, dtype=float)
    residual_vec = _as_1d_float(residual_vec)
    theta_prev = _as_1d_float(theta_prev)
    theta_low = _as_1d_float(theta_low, theta_prev.size)
    theta_high = _as_1d_float(theta_high, theta_prev.size)

    if Phi.ndim != 2 or Phi.shape[1] != theta_prev.size:
        raise ValueError("Phi must be a 2D matrix with theta_dim columns.")

    reg = (float(lambda_prev) + float(lambda_0)) * np.eye(theta_prev.size, dtype=float)
    lhs = Phi.T @ Phi + reg
    rhs = Phi.T @ residual_vec + float(lambda_prev) * theta_prev

    try:
        theta_unclipped = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        theta_unclipped = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    theta = np.clip(theta_unclipped, theta_low, theta_high)
    residual_fit = residual_vec - Phi @ theta
    return {
        "success": bool(np.all(np.isfinite(theta))),
        "solver": "ridge_closed_form",
        "theta": theta.astype(float),
        "theta_unclipped": theta_unclipped.astype(float),
        "clipped": bool(np.any(np.abs(theta - theta_unclipped) > 1e-12)),
        "residual_norm": float(np.linalg.norm(residual_fit)),
        "condition_number": float(np.linalg.cond(lhs)),
    }


def solve_bounded_least_squares(
    Phi: np.ndarray,
    residual_vec: np.ndarray,
    theta_prev: np.ndarray,
    lambda_prev: float,
    lambda_0: float,
    theta_low: np.ndarray,
    theta_high: np.ndarray,
) -> dict:
    Phi = np.asarray(Phi, dtype=float)
    residual_vec = _as_1d_float(residual_vec)
    theta_prev = _as_1d_float(theta_prev)
    theta_low = _as_1d_float(theta_low, theta_prev.size)
    theta_high = _as_1d_float(theta_high, theta_prev.size)

    sqrt_prev = float(np.sqrt(max(0.0, lambda_prev)))
    sqrt_0 = float(np.sqrt(max(0.0, lambda_0)))
    eye = np.eye(theta_prev.size, dtype=float)
    Phi_aug = np.vstack((Phi, sqrt_prev * eye, sqrt_0 * eye))
    rhs_aug = np.concatenate((residual_vec, sqrt_prev * theta_prev, np.zeros(theta_prev.size, dtype=float)))
    sol = spo.lsq_linear(Phi_aug, rhs_aug, bounds=(theta_low, theta_high), lsmr_tol="auto")
    residual_fit = residual_vec - Phi @ sol.x
    return {
        "success": bool(sol.success and np.all(np.isfinite(sol.x))),
        "solver": "bounded_least_squares",
        "theta": np.asarray(sol.x, dtype=float),
        "theta_unclipped": np.asarray(sol.x, dtype=float),
        "clipped": False,
        "residual_norm": float(np.linalg.norm(residual_fit)),
        "condition_number": float(np.linalg.cond(Phi_aug.T @ Phi_aug)),
        "status": int(sol.status),
        "message": str(sol.message),
    }


def solve_identification_batch(
    batch: RollingIDBatch,
    A0_phys: np.ndarray,
    B0_phys: np.ndarray,
    basis: dict,
    theta_prev: np.ndarray,
    cfg: dict,
) -> dict:
    Phi, residual_vec = assemble_batch_regression(batch, A0_phys=A0_phys, B0_phys=B0_phys, basis=basis)
    theta_low = _as_1d_float(cfg["theta_low"], basis["theta_dim"])
    theta_high = _as_1d_float(cfg["theta_high"], basis["theta_dim"])
    if np.any(theta_low > theta_high):
        raise ValueError("theta_low must be elementwise <= theta_high.")
    lambda_prev = float(cfg.get("lambda_prev", 1e-2))
    lambda_0 = float(cfg.get("lambda_0", 1e-4))
    solver = str(cfg.get("id_solver", "ridge_closed_form")).lower()

    if solver == "ridge_closed_form":
        result = solve_batch_ridge(
            Phi=Phi,
            residual_vec=residual_vec,
            theta_prev=theta_prev,
            lambda_prev=lambda_prev,
            lambda_0=lambda_0,
            theta_low=theta_low,
            theta_high=theta_high,
        )
    elif solver == "bounded_least_squares":
        result = solve_bounded_least_squares(
            Phi=Phi,
            residual_vec=residual_vec,
            theta_prev=theta_prev,
            lambda_prev=lambda_prev,
            lambda_0=lambda_0,
            theta_low=theta_low,
            theta_high=theta_high,
        )
    else:
        raise ValueError("id_solver must be 'ridge_closed_form' or 'bounded_least_squares'.")

    result["Phi_shape"] = tuple(Phi.shape)
    result["sample_count"] = int(batch.x_t.shape[0])
    return result


def reconstruct_model_from_theta(A0_phys: np.ndarray, B0_phys: np.ndarray, basis: dict, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A0_phys = np.asarray(A0_phys, dtype=float)
    B0_phys = np.asarray(B0_phys, dtype=float)
    theta = _as_1d_float(theta, basis["theta_dim"])

    A_out = A0_phys.copy()
    B_out = B0_phys.copy()

    offset = 0
    for E in basis["A_basis"]:
        A_out = A_out + float(theta[offset]) * np.asarray(E, dtype=float)
        offset += 1
    for F in basis["B_basis"]:
        B_out = B_out + float(theta[offset]) * np.asarray(F, dtype=float)
        offset += 1
    return A_out, B_out


def evaluate_identified_candidate(
    A_candidate: np.ndarray,
    B_candidate: np.ndarray,
    A0_phys: np.ndarray,
    B0_phys: np.ndarray,
    delta_A_max: float,
    delta_B_max: float,
) -> dict:
    A_candidate = np.asarray(A_candidate, dtype=float)
    B_candidate = np.asarray(B_candidate, dtype=float)
    finite_ok = bool(np.all(np.isfinite(A_candidate)) and np.all(np.isfinite(B_candidate)))
    A_ratio = _relative_fro(A_candidate, A0_phys)
    B_ratio = _relative_fro(B_candidate, B0_phys)
    A_ok = bool(A_ratio <= float(delta_A_max))
    B_ok = bool(B_ratio <= float(delta_B_max))
    valid = bool(finite_ok and A_ok and B_ok)
    if not finite_ok:
        reason = "nonfinite"
    elif not A_ok:
        reason = "delta_A_cap"
    elif not B_ok:
        reason = "delta_B_cap"
    else:
        reason = "accepted"
    return {
        "valid": valid,
        "finite_ok": finite_ok,
        "A_ratio": float(A_ratio),
        "B_ratio": float(B_ratio),
        "A_ok": A_ok,
        "B_ok": B_ok,
        "reason": reason,
    }


def compute_theta_clipping_diagnostics(
    theta_candidate: np.ndarray,
    theta_unclipped: np.ndarray,
    theta_low: np.ndarray,
    theta_high: np.ndarray,
    *,
    tol: float = 1e-10,
) -> dict:
    theta_candidate = _as_1d_float(theta_candidate)
    theta_unclipped = _as_1d_float(theta_unclipped, theta_candidate.size)
    theta_low = _as_1d_float(theta_low, theta_candidate.size)
    theta_high = _as_1d_float(theta_high, theta_candidate.size)
    lower_hit = np.isclose(theta_candidate, theta_low, atol=tol, rtol=0.0)
    upper_hit = np.isclose(theta_candidate, theta_high, atol=tol, rtol=0.0)
    clipped_mask = np.abs(theta_candidate - theta_unclipped) > tol
    return {
        "lower_hit_mask": lower_hit.astype(int),
        "upper_hit_mask": upper_hit.astype(int),
        "clipped_mask": clipped_mask.astype(int),
        "clipped_fraction": float(np.mean(clipped_mask.astype(float))) if clipped_mask.size > 0 else 0.0,
        "theta_within_bounds": bool(np.all(theta_candidate >= theta_low - tol) and np.all(theta_candidate <= theta_high + tol)),
    }


def select_identified_model(
    *,
    A_candidate: np.ndarray | None,
    B_candidate: np.ndarray | None,
    theta_candidate: np.ndarray | None,
    theta_unclipped: np.ndarray | None,
    solve_success: bool,
    A0_phys: np.ndarray,
    B0_phys: np.ndarray,
    theta_nominal: np.ndarray,
    A_prev: np.ndarray,
    B_prev: np.ndarray,
    theta_prev: np.ndarray,
    theta_low: np.ndarray,
    theta_high: np.ndarray,
    delta_A_max: float,
    delta_B_max: float,
    candidate_guard_mode: str = "both",
) -> dict:
    guard_mode = str(candidate_guard_mode).lower()
    if guard_mode not in {"theta_only", "fro_only", "both"}:
        raise ValueError("candidate_guard_mode must be 'theta_only', 'fro_only', or 'both'.")

    theta_low = _as_1d_float(theta_low, theta_prev.size)
    theta_high = _as_1d_float(theta_high, theta_prev.size)

    if not solve_success or A_candidate is None or B_candidate is None or theta_candidate is None:
        return {
            "A_active": np.asarray(A_prev, dtype=float),
            "B_active": np.asarray(B_prev, dtype=float),
            "theta_active": np.asarray(theta_prev, dtype=float),
            "update_success": False,
            "fallback_used": True,
            "source": "previous_or_nominal",
            "candidate_eval": {
                "valid": False,
                "finite_ok": False,
                "A_ratio": _relative_fro(A_prev, A0_phys),
                "B_ratio": _relative_fro(B_prev, B0_phys),
                "A_ok": True,
                "B_ok": True,
                "reason": "solver_failure",
                "guard_mode": guard_mode,
            },
            "theta_eval": {
                "theta_within_bounds": False,
                "guard_mode": guard_mode,
            },
        }

    theta_candidate = _as_1d_float(theta_candidate, theta_prev.size)
    theta_unclipped = theta_candidate if theta_unclipped is None else _as_1d_float(theta_unclipped, theta_prev.size)
    theta_diag = compute_theta_clipping_diagnostics(
        theta_candidate=theta_candidate,
        theta_unclipped=theta_unclipped,
        theta_low=theta_low,
        theta_high=theta_high,
    )
    theta_finite_ok = bool(np.all(np.isfinite(theta_candidate)) and np.all(np.isfinite(theta_unclipped)))
    theta_ok = bool(theta_finite_ok and theta_diag["theta_within_bounds"])

    eval_info = evaluate_identified_candidate(
        A_candidate=A_candidate,
        B_candidate=B_candidate,
        A0_phys=A0_phys,
        B0_phys=B0_phys,
        delta_A_max=delta_A_max,
        delta_B_max=delta_B_max,
    )
    theta_eval = {
        **theta_diag,
        "theta_finite_ok": theta_finite_ok,
        "theta_within_bounds": theta_diag["theta_within_bounds"],
        "theta_ok": theta_ok,
        "guard_mode": guard_mode,
    }

    if guard_mode == "theta_only":
        candidate_valid = bool(theta_ok and eval_info["finite_ok"])
        reason = "accepted" if candidate_valid else ("theta_bounds" if not theta_ok else eval_info["reason"])
    elif guard_mode == "fro_only":
        candidate_valid = bool(eval_info["valid"])
        reason = eval_info["reason"] if not candidate_valid else "accepted"
    else:
        candidate_valid = bool(theta_ok and eval_info["valid"])
        if candidate_valid:
            reason = "accepted"
        elif not theta_ok:
            reason = "theta_bounds"
        else:
            reason = eval_info["reason"]

    eval_info = {**eval_info, "valid": candidate_valid, "reason": reason, "guard_mode": guard_mode}

    if candidate_valid:
        return {
            "A_active": np.asarray(A_candidate, dtype=float),
            "B_active": np.asarray(B_candidate, dtype=float),
            "theta_active": np.asarray(theta_candidate, dtype=float),
            "update_success": True,
            "fallback_used": False,
            "source": "candidate",
            "candidate_eval": eval_info,
            "theta_eval": theta_eval,
        }

    return {
        "A_active": np.asarray(A_prev, dtype=float),
        "B_active": np.asarray(B_prev, dtype=float),
        "theta_active": np.asarray(theta_prev, dtype=float),
        "update_success": False,
        "fallback_used": True,
        "source": "previous_or_nominal",
        "candidate_eval": eval_info,
        "theta_eval": theta_eval,
    }


def map_action_to_eta(raw_action) -> tuple[float, float]:
    raw_action = float(np.asarray(raw_action, dtype=float).reshape(-1)[0])
    raw_action = float(np.clip(raw_action, -1.0, 1.0))
    eta_raw = 0.5 * (raw_action + 1.0)
    return raw_action, float(np.clip(eta_raw, 0.0, 1.0))


def eta_to_raw_action(eta: float) -> float:
    eta = float(np.clip(eta, 0.0, 1.0))
    return float(np.clip(2.0 * eta - 1.0, -1.0, 1.0))


def smooth_eta(eta_prev: float, eta_raw: float, tau_eta: float) -> float:
    tau_eta = float(np.clip(tau_eta, 0.0, 1.0))
    return float((1.0 - tau_eta) * float(eta_prev) + tau_eta * float(eta_raw))


def blend_prediction_model(A0_phys: np.ndarray, B0_phys: np.ndarray, A_id_phys: np.ndarray, B_id_phys: np.ndarray, eta: float) -> tuple[np.ndarray, np.ndarray]:
    eta = float(np.clip(eta, 0.0, 1.0))
    A0_phys = np.asarray(A0_phys, dtype=float)
    B0_phys = np.asarray(B0_phys, dtype=float)
    A_id_phys = np.asarray(A_id_phys, dtype=float)
    B_id_phys = np.asarray(B_id_phys, dtype=float)
    A_pred = A0_phys + eta * (A_id_phys - A0_phys)
    B_pred = B0_phys + eta * (B_id_phys - B0_phys)
    return A_pred, B_pred


def build_blend_policy_state(
    base_state,
    prev_eta: float,
    residual_norm: float,
    A_ratio: float,
    B_ratio: float,
    id_valid_flag: float,
) -> np.ndarray:
    base_state = _as_1d_float(base_state)
    extras = np.asarray(
        [
            float(prev_eta),
            float(residual_norm),
            float(A_ratio),
            float(B_ratio),
            float(id_valid_flag),
        ],
        dtype=np.float32,
    )
    return np.concatenate((base_state.astype(np.float32), extras), axis=0).astype(np.float32)


def build_blend_policy_state_v2(
    base_state,
    *,
    prev_eta: float,
    residual_norm: float,
    active_A_ratio: float,
    active_B_ratio: float,
    candidate_A_ratio: float,
    candidate_B_ratio: float,
    id_valid_flag: float,
    id_update_success_flag: float,
    id_fallback_flag: float,
    delta_A_max: float,
    delta_B_max: float,
    normalize_blend_extras: bool = True,
    blend_extra_clip: float = 3.0,
    blend_residual_scale: float = 1.0,
) -> np.ndarray:
    base_state = _as_1d_float(base_state)
    if normalize_blend_extras:
        eps = 1e-8
        z_eta = float(np.clip(2.0 * float(np.clip(prev_eta, 0.0, 1.0)) - 1.0, -1.0, 1.0))
        residual_scale = max(float(blend_residual_scale), eps)
        z_r = float(
            np.clip(
                np.log1p(max(0.0, float(residual_norm))) / residual_scale,
                -float(blend_extra_clip),
                float(blend_extra_clip),
            )
        )
        z_A = float(np.clip(float(active_A_ratio) / max(float(delta_A_max), eps), 0.0, 1.5))
        z_B = float(np.clip(float(active_B_ratio) / max(float(delta_B_max), eps), 0.0, 1.5))
        z_A_cand = float(np.clip(float(candidate_A_ratio) / max(float(delta_A_max), eps), 0.0, 1.5))
        z_B_cand = float(np.clip(float(candidate_B_ratio) / max(float(delta_B_max), eps), 0.0, 1.5))
        extras = np.asarray(
            [
                z_eta,
                z_r,
                z_A,
                z_B,
                z_A_cand,
                z_B_cand,
                float(id_valid_flag),
                float(id_update_success_flag),
                float(id_fallback_flag),
            ],
            dtype=np.float32,
        )
    else:
        extras = np.asarray(
            [
                float(prev_eta),
                float(residual_norm),
                float(active_A_ratio),
                float(active_B_ratio),
                float(candidate_A_ratio),
                float(candidate_B_ratio),
                float(id_valid_flag),
                float(id_update_success_flag),
                float(id_fallback_flag),
            ],
            dtype=np.float32,
        )
    return np.concatenate((base_state.astype(np.float32), extras), axis=0).astype(np.float32)


def get_blend_state_dim(base_aug_dim: int, n_outputs: int, n_inputs: int, state_mode: str) -> int:
    return int(get_rl_state_dim(base_aug_dim, n_outputs, n_inputs, state_mode)) + 5


def get_blend_state_dim_v2(base_aug_dim: int, n_outputs: int, n_inputs: int, state_mode: str) -> int:
    return int(get_rl_state_dim(base_aug_dim, n_outputs, n_inputs, state_mode)) + 9
