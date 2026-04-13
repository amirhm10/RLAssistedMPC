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
    n_A = len(A_basis)
    n_B = len(B_basis)
    if len(theta_labels) != n_A + n_B:
        raise ValueError("theta_labels must match the total number of A and B basis elements.")
    theta_A_indices = np.arange(n_A, dtype=int)
    theta_B_indices = np.arange(n_A, n_A + n_B, dtype=int)
    return {
        "basis_name": basis_name,
        "basis_family": basis_family,
        "A_basis": [np.asarray(E, dtype=float) for E in A_basis],
        "B_basis": [np.asarray(F, dtype=float) for F in B_basis],
        "theta_labels": list(theta_labels),
        "theta_labels_A": list(theta_labels[:n_A]),
        "theta_labels_B": list(theta_labels[n_A:]),
        "theta_A_indices": theta_A_indices,
        "theta_B_indices": theta_B_indices,
        "n_A": n_A,
        "n_B": n_B,
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


def _expand_theta_vector(value, size: int, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == int(size):
        return arr.astype(float)
    if arr.size == 1:
        return np.full(int(size), float(arr[0]), dtype=float)
    raise ValueError(f"{name} must be length 1 or {size}, got {arr.size}.")


def resolve_identification_theta_bounds(basis: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    theta_dim = int(basis["theta_dim"])
    theta_A_indices = np.asarray(basis["theta_A_indices"], dtype=int)
    theta_B_indices = np.asarray(basis["theta_B_indices"], dtype=int)
    has_split_bounds = any(
        key in cfg for key in ("theta_low_A", "theta_high_A", "theta_low_B", "theta_high_B")
    )

    if "theta_low" in cfg:
        theta_low = _expand_theta_vector(cfg["theta_low"], theta_dim, name="theta_low")
    elif has_split_bounds:
        theta_low = np.full(theta_dim, -np.inf, dtype=float)
    else:
        raise KeyError("cfg must contain theta_low/theta_high or split theta bounds.")

    if "theta_high" in cfg:
        theta_high = _expand_theta_vector(cfg["theta_high"], theta_dim, name="theta_high")
    elif has_split_bounds:
        theta_high = np.full(theta_dim, np.inf, dtype=float)
    else:
        raise KeyError("cfg must contain theta_low/theta_high or split theta bounds.")

    if "theta_low_A" in cfg:
        theta_low[theta_A_indices] = _expand_theta_vector(
            cfg["theta_low_A"],
            theta_A_indices.size,
            name="theta_low_A",
        )
    if "theta_high_A" in cfg:
        theta_high[theta_A_indices] = _expand_theta_vector(
            cfg["theta_high_A"],
            theta_A_indices.size,
            name="theta_high_A",
        )
    if "theta_low_B" in cfg:
        theta_low[theta_B_indices] = _expand_theta_vector(
            cfg["theta_low_B"],
            theta_B_indices.size,
            name="theta_low_B",
        )
    if "theta_high_B" in cfg:
        theta_high[theta_B_indices] = _expand_theta_vector(
            cfg["theta_high_B"],
            theta_B_indices.size,
            name="theta_high_B",
        )

    if np.any(theta_low > theta_high):
        raise ValueError("theta_low must be elementwise <= theta_high.")
    return theta_low.astype(float), theta_high.astype(float)


def resolve_identification_lambda_vectors(basis: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    theta_dim = int(basis["theta_dim"])
    theta_A_indices = np.asarray(basis["theta_A_indices"], dtype=int)
    theta_B_indices = np.asarray(basis["theta_B_indices"], dtype=int)

    lambda_prev_vec = np.full(theta_dim, float(cfg.get("lambda_prev", 1e-2)), dtype=float)
    lambda_0_vec = np.full(theta_dim, float(cfg.get("lambda_0", 1e-4)), dtype=float)

    if "lambda_prev_A" in cfg:
        lambda_prev_vec[theta_A_indices] = _expand_theta_vector(
            cfg["lambda_prev_A"],
            theta_A_indices.size,
            name="lambda_prev_A",
        )
    if "lambda_prev_B" in cfg:
        lambda_prev_vec[theta_B_indices] = _expand_theta_vector(
            cfg["lambda_prev_B"],
            theta_B_indices.size,
            name="lambda_prev_B",
        )
    if "lambda_0_A" in cfg:
        lambda_0_vec[theta_A_indices] = _expand_theta_vector(
            cfg["lambda_0_A"],
            theta_A_indices.size,
            name="lambda_0_A",
        )
    if "lambda_0_B" in cfg:
        lambda_0_vec[theta_B_indices] = _expand_theta_vector(
            cfg["lambda_0_B"],
            theta_B_indices.size,
            name="lambda_0_B",
        )
    return lambda_prev_vec.astype(float), lambda_0_vec.astype(float)


def resolve_identification_component_indices(basis: dict, id_component_mode: str = "AB") -> tuple[np.ndarray, np.ndarray]:
    mode = str(id_component_mode).upper()
    theta_dim = int(basis["theta_dim"])
    theta_A_indices = np.asarray(basis["theta_A_indices"], dtype=int)
    theta_B_indices = np.asarray(basis["theta_B_indices"], dtype=int)
    if mode == "AB":
        active = np.arange(theta_dim, dtype=int)
    elif mode == "A_ONLY":
        active = theta_A_indices.copy()
    elif mode == "B_ONLY":
        active = theta_B_indices.copy()
    else:
        raise ValueError("id_component_mode must be 'AB', 'A_only', or 'B_only'.")
    inactive = np.setdiff1d(np.arange(theta_dim, dtype=int), active, assume_unique=True)
    return active.astype(int), inactive.astype(int)


def solve_batch_ridge(
    Phi: np.ndarray,
    residual_vec: np.ndarray,
    theta_prev: np.ndarray,
    lambda_prev,
    lambda_0,
    theta_low: np.ndarray,
    theta_high: np.ndarray,
) -> dict:
    Phi = np.asarray(Phi, dtype=float)
    residual_vec = _as_1d_float(residual_vec)
    theta_prev = _as_1d_float(theta_prev)
    theta_low = _as_1d_float(theta_low, theta_prev.size)
    theta_high = _as_1d_float(theta_high, theta_prev.size)
    lambda_prev_vec = _expand_theta_vector(lambda_prev, theta_prev.size, name="lambda_prev")
    lambda_0_vec = _expand_theta_vector(lambda_0, theta_prev.size, name="lambda_0")

    if Phi.ndim != 2 or Phi.shape[1] != theta_prev.size:
        raise ValueError("Phi must be a 2D matrix with theta_dim columns.")

    reg = np.diag(lambda_prev_vec + lambda_0_vec)
    lhs = Phi.T @ Phi + reg
    rhs = Phi.T @ residual_vec + lambda_prev_vec * theta_prev

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
    lambda_prev,
    lambda_0,
    theta_low: np.ndarray,
    theta_high: np.ndarray,
) -> dict:
    Phi = np.asarray(Phi, dtype=float)
    residual_vec = _as_1d_float(residual_vec)
    theta_prev = _as_1d_float(theta_prev)
    theta_low = _as_1d_float(theta_low, theta_prev.size)
    theta_high = _as_1d_float(theta_high, theta_prev.size)
    lambda_prev_vec = _expand_theta_vector(lambda_prev, theta_prev.size, name="lambda_prev")
    lambda_0_vec = _expand_theta_vector(lambda_0, theta_prev.size, name="lambda_0")

    reg_prev = np.diag(np.sqrt(np.clip(lambda_prev_vec, 0.0, None)))
    reg_0 = np.diag(np.sqrt(np.clip(lambda_0_vec, 0.0, None)))
    Phi_aug = np.vstack((Phi, reg_prev, reg_0))
    rhs_aug = np.concatenate((residual_vec, reg_prev @ theta_prev, np.zeros(theta_prev.size, dtype=float)))
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
    theta_prev = _as_1d_float(theta_prev, basis["theta_dim"])
    theta_low, theta_high = resolve_identification_theta_bounds(basis=basis, cfg=cfg)
    lambda_prev_vec, lambda_0_vec = resolve_identification_lambda_vectors(basis=basis, cfg=cfg)
    id_component_mode = str(cfg.get("id_component_mode", "AB"))
    active_indices, inactive_indices = resolve_identification_component_indices(basis=basis, id_component_mode=id_component_mode)
    Phi_active = Phi[:, active_indices]
    theta_prev_active = theta_prev[active_indices]
    theta_low_active = theta_low[active_indices]
    theta_high_active = theta_high[active_indices]
    lambda_prev_active = lambda_prev_vec[active_indices]
    lambda_0_active = lambda_0_vec[active_indices]
    solver = str(cfg.get("id_solver", "ridge_closed_form")).lower()

    if solver == "ridge_closed_form":
        result = solve_batch_ridge(
            Phi=Phi_active,
            residual_vec=residual_vec,
            theta_prev=theta_prev_active,
            lambda_prev=lambda_prev_active,
            lambda_0=lambda_0_active,
            theta_low=theta_low_active,
            theta_high=theta_high_active,
        )
    elif solver == "bounded_least_squares":
        result = solve_bounded_least_squares(
            Phi=Phi_active,
            residual_vec=residual_vec,
            theta_prev=theta_prev_active,
            lambda_prev=lambda_prev_active,
            lambda_0=lambda_0_active,
            theta_low=theta_low_active,
            theta_high=theta_high_active,
        )
    else:
        raise ValueError("id_solver must be 'ridge_closed_form' or 'bounded_least_squares'.")

    theta_full = np.zeros(basis["theta_dim"], dtype=float)
    theta_unclipped_full = np.zeros(basis["theta_dim"], dtype=float)
    theta_full[active_indices] = np.asarray(result["theta"], float)
    theta_unclipped_full[active_indices] = np.asarray(result["theta_unclipped"], float)
    if inactive_indices.size > 0:
        theta_full[inactive_indices] = 0.0
        theta_unclipped_full[inactive_indices] = 0.0
    result["theta"] = theta_full
    result["theta_unclipped"] = theta_unclipped_full
    result["Phi_shape"] = tuple(Phi.shape)
    result["Phi_active_shape"] = tuple(Phi_active.shape)
    result["sample_count"] = int(batch.x_t.shape[0])
    result["theta_low"] = theta_low
    result["theta_high"] = theta_high
    result["lambda_prev_vec"] = lambda_prev_vec
    result["lambda_0_vec"] = lambda_0_vec
    result["theta_active_indices"] = active_indices
    result["theta_inactive_indices"] = inactive_indices
    result["id_component_mode"] = str(id_component_mode)
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


def summarize_theta_clipping_statistics(
    *,
    theta_active_log,
    theta_candidate_log=None,
    theta_unclipped_log=None,
    theta_clipped_fraction_log=None,
    theta_A_indices=None,
    theta_B_indices=None,
    tail_window: int | None = None,
) -> dict:
    theta_active = np.asarray(theta_active_log, dtype=float)
    if theta_active.ndim != 2:
        raise ValueError("theta_active_log must be a 2D array.")
    n_steps = int(theta_active.shape[0])
    if n_steps == 0:
        return {
            "tail_window": 0,
            "mean_abs_theta_active_tail": 0.0,
            "mean_abs_theta_A_active_tail": 0.0,
            "mean_abs_theta_B_active_tail": 0.0,
            "clipping_fraction_mean": 0.0,
            "theta_candidate_delta_mean": 0.0,
            "theta_unclipped_delta_mean": 0.0,
        }
    tail_window = n_steps if tail_window is None else max(1, min(int(tail_window), n_steps))
    theta_tail = theta_active[-tail_window:, :]
    theta_A_indices = np.asarray([] if theta_A_indices is None else theta_A_indices, dtype=int).reshape(-1)
    theta_B_indices = np.asarray([] if theta_B_indices is None else theta_B_indices, dtype=int).reshape(-1)

    stats = {
        "tail_window": int(tail_window),
        "mean_abs_theta_active_tail": float(np.mean(np.abs(theta_tail))),
        "mean_abs_theta_A_active_tail": 0.0,
        "mean_abs_theta_B_active_tail": 0.0,
        "clipping_fraction_mean": 0.0,
        "theta_candidate_delta_mean": 0.0,
        "theta_unclipped_delta_mean": 0.0,
    }
    if theta_A_indices.size > 0:
        stats["mean_abs_theta_A_active_tail"] = float(np.mean(np.abs(theta_tail[:, theta_A_indices])))
    if theta_B_indices.size > 0:
        stats["mean_abs_theta_B_active_tail"] = float(np.mean(np.abs(theta_tail[:, theta_B_indices])))
    if theta_clipped_fraction_log is not None:
        clipped = np.asarray(theta_clipped_fraction_log, dtype=float).reshape(-1)
        stats["clipping_fraction_mean"] = float(np.mean(clipped[-tail_window:]))
    if theta_candidate_log is not None:
        theta_candidate = np.asarray(theta_candidate_log, dtype=float)
        if theta_candidate.shape == theta_active.shape:
            stats["theta_candidate_delta_mean"] = float(
                np.mean(np.abs(theta_candidate[-tail_window:, :] - theta_tail))
            )
    if theta_unclipped_log is not None:
        theta_unclipped = np.asarray(theta_unclipped_log, dtype=float)
        if theta_unclipped.shape == theta_active.shape:
            stats["theta_unclipped_delta_mean"] = float(
                np.mean(np.abs(theta_unclipped[-tail_window:, :] - theta_tail))
            )
    return stats


def summarize_reid_run_statistics(result_bundle: dict, *, tail_window: int | None = None) -> dict:
    rewards = np.asarray(result_bundle.get("rewards_step", []), dtype=float).reshape(-1)
    eta_log = np.asarray(result_bundle.get("eta_log", []), dtype=float).reshape(-1)
    active_A_ratio = np.asarray(result_bundle.get("active_A_model_delta_ratio_log", []), dtype=float).reshape(-1)
    active_B_ratio = np.asarray(result_bundle.get("active_B_model_delta_ratio_log", []), dtype=float).reshape(-1)
    theta_active_log = np.asarray(result_bundle.get("theta_active_log", []), dtype=float)
    theta_candidate_log = result_bundle.get("theta_candidate_log")
    theta_unclipped_log = result_bundle.get("theta_unclipped_log")
    theta_clipped_fraction_log = result_bundle.get("theta_clipped_fraction_log")
    theta_A_indices = result_bundle.get("theta_A_indices")
    theta_B_indices = result_bundle.get("theta_B_indices")
    id_fallback_log = np.asarray(result_bundle.get("id_fallback_log", []), dtype=float).reshape(-1)
    id_update_success_log = np.asarray(result_bundle.get("id_update_success_log", []), dtype=float).reshape(-1)
    id_update_event_log = np.asarray(result_bundle.get("id_update_event_log", []), dtype=float).reshape(-1)

    n_steps = int(rewards.size)
    tail_window = n_steps if tail_window is None else max(1, min(int(tail_window), n_steps if n_steps > 0 else 1))
    tail_slice = slice(max(0, n_steps - tail_window), n_steps)

    stats = {
        "evaluation_window_steps": int(tail_window if n_steps > 0 else 0),
        "final_reward": float(rewards[-1]) if rewards.size > 0 else 0.0,
        "best_reward": float(np.max(rewards)) if rewards.size > 0 else 0.0,
        "tail_reward_mean": float(np.mean(rewards[tail_slice])) if rewards.size > 0 else 0.0,
        "tail_eta_mean": float(np.mean(eta_log[tail_slice])) if eta_log.size > 0 else 0.0,
        "tail_active_A_ratio_mean": float(np.mean(active_A_ratio[tail_slice])) if active_A_ratio.size > 0 else 0.0,
        "tail_active_B_ratio_mean": float(np.mean(active_B_ratio[tail_slice])) if active_B_ratio.size > 0 else 0.0,
        "fallback_fraction": 0.0,
        "update_success_fraction": 0.0,
        "update_event_count": int(np.sum(id_update_event_log)) if id_update_event_log.size > 0 else 0,
        "update_success_count": int(result_bundle.get("id_update_success_count", int(np.sum(id_update_success_log)))),
        "invalid_solve_count": int(result_bundle.get("invalid_id_solve_count", 0)),
        "id_solver_failure_count": int(result_bundle.get("id_solver_failure_count", 0)),
    }
    if id_update_event_log.size > 0 and np.any(id_update_event_log > 0.5):
        event_mask = id_update_event_log > 0.5
        stats["fallback_fraction"] = float(np.mean(id_fallback_log[event_mask])) if id_fallback_log.size > 0 else 0.0
        stats["update_success_fraction"] = (
            float(np.mean(id_update_success_log[event_mask])) if id_update_success_log.size > 0 else 0.0
        )
    elif id_fallback_log.size > 0:
        stats["fallback_fraction"] = float(np.mean(id_fallback_log))
        stats["update_success_fraction"] = (
            float(np.mean(id_update_success_log)) if id_update_success_log.size > 0 else 0.0
        )

    if theta_active_log.size > 0:
        stats.update(
            summarize_theta_clipping_statistics(
                theta_active_log=theta_active_log,
                theta_candidate_log=theta_candidate_log,
                theta_unclipped_log=theta_unclipped_log,
                theta_clipped_fraction_log=theta_clipped_fraction_log,
                theta_A_indices=theta_A_indices,
                theta_B_indices=theta_B_indices,
                tail_window=tail_window,
            )
        )
    return stats
