from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.optimize as spo

from Simulation.mpc import augment_state_space
from utils.helpers import apply_min_max
from utils.observer import compute_observer_gain
from utils.state_features import get_rl_state_dim


REIDENTIFICATION_BASIS_FAMILY = "lowrank_polymer"
REIDENTIFICATION_COMPONENT_MODE = "AB"
REIDENTIFICATION_OBSERVER_ALIGNMENT = "legacy_previous_measurement"
REIDENTIFICATION_CANDIDATE_GUARD_MODE = "fro_only"
REIDENTIFICATION_NORMALIZE_BLEND_EXTRAS = True
REIDENTIFICATION_BLEND_EXTRA_CLIP = 3.0
REIDENTIFICATION_BLEND_RESIDUAL_SCALE = 1.0
REIDENTIFICATION_LOG_THETA_CLIPPING = True
REIDENTIFICATION_GUARD_VALIDATION_FRACTION = 0.0
REIDENTIFICATION_GUARD_MIN_VALIDATION_SAMPLES = 0
REIDENTIFICATION_GUARD_MIN_TRAIN_SAMPLES = 1
REIDENTIFICATION_GUARD_MAX_THETA_CLIPPED_FRACTION = 1.0
REIDENTIFICATION_GUARD_MAX_CONDITION_NUMBER = np.inf
REIDENTIFICATION_GUARD_MAX_VALIDATION_RESIDUAL_RATIO = 1.0
REIDENTIFICATION_GUARD_MAX_FULL_RESIDUAL_RATIO = 1.0
REIDENTIFICATION_BLEND_VALIDITY_MODE = "off"
REIDENTIFICATION_BLEND_VALIDITY_SCALE_FLOOR = 0.0
REIDENTIFICATION_BLEND_VALIDITY_RESIDUAL_SOFT = np.inf
REIDENTIFICATION_BLEND_VALIDITY_RESIDUAL_HARD = np.inf
REIDENTIFICATION_BLEND_VALIDITY_CLIPPED_SOFT = 1.0
REIDENTIFICATION_BLEND_VALIDITY_CLIPPED_HARD = 1.0
REIDENTIFICATION_BLEND_VALIDITY_CONDITION_SOFT = np.inf
REIDENTIFICATION_BLEND_VALIDITY_CONDITION_HARD = np.inf
REIDENTIFICATION_BLEND_VALIDITY_FALLBACK_SCALE = 1.0
REIDENTIFICATION_BLEND_VALIDITY_INVALID_CANDIDATE_SCALE = 1.0


def resolve_basis_family(basis_family=None) -> str:
    family = str(REIDENTIFICATION_BASIS_FAMILY if basis_family in (None, "") else basis_family).strip().lower()
    if not family:
        raise ValueError("basis_family must be a non-empty string.")
    return family


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


def _vectorize_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=float).reshape(-1, order="F")


def _unvectorize_matrix(vector: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(vector, dtype=float).reshape(shape, order="F")


def _svd_directions(matrix_stack: np.ndarray, target_rank: int) -> tuple[list[np.ndarray], np.ndarray]:
    matrix_stack = np.asarray(matrix_stack, dtype=float)
    if matrix_stack.ndim != 2:
        raise ValueError("matrix_stack must be 2D.")
    target_rank = int(target_rank)
    if target_rank <= 0:
        raise ValueError("target_rank must be positive.")
    if matrix_stack.shape[1] == 0:
        raise ValueError("matrix_stack must contain at least one sample column.")

    U, singular_values, _ = np.linalg.svd(matrix_stack, full_matrices=False)
    actual_rank = max(1, min(target_rank, U.shape[1]))
    return [np.asarray(U[:, idx], dtype=float) for idx in range(actual_rank)], singular_values.astype(float)


def _make_basis_dict(
    *,
    basis_name: str,
    basis_family: str,
    A_basis: list[np.ndarray],
    B_basis: list[np.ndarray],
    singular_values_A: np.ndarray,
    singular_values_B: np.ndarray,
    metadata: dict,
) -> dict:
    n_A = len(A_basis)
    n_B = len(B_basis)
    theta_labels_A = [f"alpha_{idx + 1}" for idx in range(n_A)]
    theta_labels_B = [f"beta_{idx + 1}" for idx in range(n_B)]
    theta_labels = theta_labels_A + theta_labels_B
    theta_A_indices = np.arange(n_A, dtype=int)
    theta_B_indices = np.arange(n_A, n_A + n_B, dtype=int)
    return {
        "basis_name": basis_name,
        "basis_family": resolve_basis_family(basis_family),
        "A_basis": [np.asarray(E, dtype=float) for E in A_basis],
        "B_basis": [np.asarray(F, dtype=float) for F in B_basis],
        "theta_labels": list(theta_labels),
        "theta_labels_A": list(theta_labels_A),
        "theta_labels_B": list(theta_labels_B),
        "alpha_labels": list(theta_labels_A),
        "beta_labels": list(theta_labels_B),
        "theta_A_indices": theta_A_indices,
        "theta_B_indices": theta_B_indices,
        "n_A": n_A,
        "n_B": n_B,
        "theta_dim": len(theta_labels),
        "singular_values_A": np.asarray(singular_values_A, dtype=float),
        "singular_values_B": np.asarray(singular_values_B, dtype=float),
        "metadata": dict(metadata),
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


def _resolve_window_slices(n_samples: int, window: int, stride: int) -> list[slice]:
    n_samples = int(n_samples)
    window = int(window)
    stride = int(stride)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive.")
    if n_samples <= window:
        return [slice(0, n_samples)]

    starts = list(range(0, n_samples - window + 1, stride))
    if starts[-1] != n_samples - window:
        starts.append(n_samples - window)
    return [slice(start, start + window) for start in starts]


def solve_dense_local_residual_fit(
    *,
    x_t: np.ndarray,
    u_t: np.ndarray,
    x_tp1: np.ndarray,
    A_ref: np.ndarray,
    B_ref: np.ndarray,
    lambda_A_off: float,
    lambda_B_off: float,
) -> dict:
    x_t = np.asarray(x_t, dtype=float)
    u_t = np.asarray(u_t, dtype=float)
    x_tp1 = np.asarray(x_tp1, dtype=float)
    A_ref, B_ref = _validate_physical_model_shapes(A_ref, B_ref)
    if x_t.ndim != 2 or x_tp1.ndim != 2 or u_t.ndim != 2:
        raise ValueError("x_t, u_t, and x_tp1 must be 2D arrays.")
    if x_t.shape != x_tp1.shape:
        raise ValueError("x_t and x_tp1 must have the same shape.")
    if x_t.shape[0] != u_t.shape[0]:
        raise ValueError("x_t and u_t must have the same number of samples.")
    if x_t.shape[1] != A_ref.shape[0] or u_t.shape[1] != B_ref.shape[1]:
        raise ValueError("Offline fit inputs do not match the reference model shape.")

    Y = x_tp1 - x_t @ A_ref.T - u_t @ B_ref.T
    X = np.concatenate((x_t, u_t), axis=1)
    n_phys = int(A_ref.shape[0])
    reg_diag = np.concatenate(
        (
            np.full(n_phys, float(lambda_A_off), dtype=float),
            np.full(B_ref.shape[1], float(lambda_B_off), dtype=float),
        ),
        axis=0,
    )
    lhs = X.T @ X + np.diag(reg_diag)
    rhs = X.T @ Y
    try:
        coeff = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coeff = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    delta_A = coeff[:n_phys, :].T
    delta_B = coeff[n_phys:, :].T
    fit = X @ coeff
    residual = Y - fit
    return {
        "delta_A": np.asarray(delta_A, dtype=float),
        "delta_B": np.asarray(delta_B, dtype=float),
        "residual_norm": float(np.linalg.norm(residual)),
        "condition_number": float(np.linalg.cond(lhs)),
        "sample_count": int(x_t.shape[0]),
    }


def _extract_offline_baseline_signals(
    *,
    baseline_bundle: dict,
    n_phys: int,
    n_inputs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xhatdhat = np.asarray(baseline_bundle["xhatdhat"], dtype=float)
    if xhatdhat.ndim != 2 or xhatdhat.shape[0] < n_phys:
        raise ValueError("Baseline bundle xhatdhat does not contain enough physical states.")

    u_abs = baseline_bundle.get("u_mpc", baseline_bundle.get("u"))
    if u_abs is None:
        raise KeyError("Baseline bundle must contain u_mpc or u.")
    u_abs = np.asarray(u_abs, dtype=float)
    if u_abs.ndim != 2 or u_abs.shape[1] != n_inputs:
        raise ValueError("Baseline bundle input array has the wrong shape.")

    steady_states = baseline_bundle["steady_states"]
    data_min = np.asarray(baseline_bundle["data_min"], dtype=float)
    data_max = np.asarray(baseline_bundle["data_max"], dtype=float)
    ss_scaled_inputs = apply_min_max(np.asarray(steady_states["ss_inputs"], dtype=float), data_min[:n_inputs], data_max[:n_inputs])
    u_scaled_abs = apply_min_max(u_abs, data_min[:n_inputs], data_max[:n_inputs])
    u_dev = u_scaled_abs - ss_scaled_inputs

    x_phys = xhatdhat[:n_phys, :].T
    n_samples = int(min(len(u_dev), max(0, x_phys.shape[0] - 1)))
    if n_samples <= 0:
        raise ValueError("Baseline bundle does not contain enough samples for offline basis extraction.")
    return x_phys[:n_samples, :], u_dev[:n_samples, :], x_phys[1 : n_samples + 1, :]


def extract_lowrank_residual_basis_from_baseline(
    *,
    baseline_bundle: dict,
    A_ref: np.ndarray,
    B_ref: np.ndarray,
    basis_family: str | None = None,
    rank_A: int,
    rank_B: int,
    offline_window: int,
    offline_stride: int,
    lambda_A_off: float,
    lambda_B_off: float,
) -> dict:
    A_ref, B_ref = _validate_physical_model_shapes(A_ref, B_ref)
    n_phys = int(A_ref.shape[0])
    n_inputs = int(B_ref.shape[1])
    x_t, u_t, x_tp1 = _extract_offline_baseline_signals(
        baseline_bundle=baseline_bundle,
        n_phys=n_phys,
        n_inputs=n_inputs,
    )

    delta_A_cols = []
    delta_B_cols = []
    fit_residual_norms = []
    fit_condition_numbers = []
    window_slices = _resolve_window_slices(len(x_t), offline_window, offline_stride)
    for window_slice in window_slices:
        fit = solve_dense_local_residual_fit(
            x_t=x_t[window_slice, :],
            u_t=u_t[window_slice, :],
            x_tp1=x_tp1[window_slice, :],
            A_ref=A_ref,
            B_ref=B_ref,
            lambda_A_off=lambda_A_off,
            lambda_B_off=lambda_B_off,
        )
        delta_A_cols.append(_vectorize_matrix(fit["delta_A"]))
        delta_B_cols.append(_vectorize_matrix(fit["delta_B"]))
        fit_residual_norms.append(float(fit["residual_norm"]))
        fit_condition_numbers.append(float(fit["condition_number"]))

    A_vectors, singular_values_A = _svd_directions(np.column_stack(delta_A_cols), rank_A)
    B_vectors, singular_values_B = _svd_directions(np.column_stack(delta_B_cols), rank_B)
    A_basis = [_unvectorize_matrix(vec, A_ref.shape) for vec in A_vectors]
    B_basis = [_unvectorize_matrix(vec, B_ref.shape) for vec in B_vectors]
    metadata = {
        "rank_A": int(len(A_basis)),
        "rank_B": int(len(B_basis)),
        "requested_rank_A": int(rank_A),
        "requested_rank_B": int(rank_B),
        "offline_window": int(offline_window),
        "offline_stride": int(offline_stride),
        "lambda_A_off": float(lambda_A_off),
        "lambda_B_off": float(lambda_B_off),
        "window_count": int(len(window_slices)),
        "fit_residual_norms": np.asarray(fit_residual_norms, dtype=float),
        "fit_condition_numbers": np.asarray(fit_condition_numbers, dtype=float),
        "A_shape": tuple(int(v) for v in A_ref.shape),
        "B_shape": tuple(int(v) for v in B_ref.shape),
    }
    basis_family = resolve_basis_family(basis_family)
    basis_slug = basis_family.replace("lowrank_", "")
    return _make_basis_dict(
        basis_name=f"{basis_slug}_reidentification_lowrank_rA{len(A_basis)}_rB{len(B_basis)}",
        basis_family=basis_family,
        A_basis=A_basis,
        B_basis=B_basis,
        singular_values_A=singular_values_A,
        singular_values_B=singular_values_B,
        metadata=metadata,
    )


def resolve_lowrank_basis_cache_path(
    cache_dir,
    *,
    basis_family: str | None = None,
    run_mode: str,
    disturbance_profile: str | None = None,
    rank_A: int,
    rank_B: int,
) -> Path:
    basis_family = resolve_basis_family(basis_family)
    run_mode = str(run_mode).lower()
    if basis_family == "lowrank_distillation":
        disturbance_profile = str("none" if disturbance_profile in (None, "") else disturbance_profile).lower()
        fname = (
            f"reidentification_lowrank_basis_distillation_{run_mode}_{disturbance_profile}"
            f"_rA{int(rank_A)}_rB{int(rank_B)}.pickle"
        )
    else:
        fname = f"reidentification_lowrank_basis_{run_mode}_rA{int(rank_A)}_rB{int(rank_B)}.pickle"
    return Path(cache_dir) / fname


def _validate_cached_basis(
    basis: dict,
    *,
    basis_family: str | None,
    A_ref: np.ndarray,
    B_ref: np.ndarray,
    baseline_path: Path,
    disturbance_profile: str | None,
    rank_A: int,
    rank_B: int,
    offline_window: int,
    offline_stride: int,
    lambda_A_off: float,
    lambda_B_off: float,
) -> bool:
    try:
        metadata = dict(basis.get("metadata") or {})
        if str(basis.get("basis_family")) != resolve_basis_family(basis_family):
            return False
        if int(metadata.get("requested_rank_A", -1)) != int(rank_A):
            return False
        if int(metadata.get("requested_rank_B", -1)) != int(rank_B):
            return False
        if int(metadata.get("offline_window", -1)) != int(offline_window):
            return False
        if int(metadata.get("offline_stride", -1)) != int(offline_stride):
            return False
        if float(metadata.get("lambda_A_off", np.nan)) != float(lambda_A_off):
            return False
        if float(metadata.get("lambda_B_off", np.nan)) != float(lambda_B_off):
            return False
        if tuple(metadata.get("A_shape") or ()) != tuple(int(v) for v in A_ref.shape):
            return False
        if tuple(metadata.get("B_shape") or ()) != tuple(int(v) for v in B_ref.shape):
            return False
        if Path(metadata.get("source_path", "")) != Path(baseline_path):
            return False
        if int(metadata.get("source_mtime_ns", -1)) != int(baseline_path.stat().st_mtime_ns):
            return False
        if disturbance_profile is not None and str(metadata.get("disturbance_profile", "")).lower() != str(disturbance_profile).lower():
            return False
        if len(basis.get("A_basis") or []) <= 0 or len(basis.get("B_basis") or []) <= 0:
            return False
    except Exception:
        return False
    return True


def build_or_load_reidentification_basis(
    *,
    baseline_path,
    cache_dir,
    A_ref: np.ndarray,
    B_ref: np.ndarray,
    basis_family: str | None = None,
    rank_A: int,
    rank_B: int,
    offline_window: int,
    offline_stride: int,
    lambda_A_off: float,
    lambda_B_off: float,
    run_mode: str,
    disturbance_profile: str | None = None,
) -> dict:
    basis_family = resolve_basis_family(basis_family)
    baseline_path = Path(baseline_path).expanduser().resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Could not find baseline bundle for offline basis extraction: {baseline_path}")
    cache_path = resolve_lowrank_basis_cache_path(
        cache_dir,
        basis_family=basis_family,
        run_mode=run_mode,
        disturbance_profile=disturbance_profile,
        rank_A=rank_A,
        rank_B=rank_B,
    )
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as handle:
                cached_basis = pickle.load(handle)
            if _validate_cached_basis(
                cached_basis,
                basis_family=basis_family,
                A_ref=A_ref,
                B_ref=B_ref,
                baseline_path=baseline_path,
                disturbance_profile=disturbance_profile,
                rank_A=rank_A,
                rank_B=rank_B,
                offline_window=offline_window,
                offline_stride=offline_stride,
                lambda_A_off=lambda_A_off,
                lambda_B_off=lambda_B_off,
            ):
                return cached_basis
        except Exception:
            pass

    with open(baseline_path, "rb") as handle:
        baseline_bundle = pickle.load(handle)
    basis = extract_lowrank_residual_basis_from_baseline(
        baseline_bundle=baseline_bundle,
        A_ref=A_ref,
        B_ref=B_ref,
        basis_family=basis_family,
        rank_A=rank_A,
        rank_B=rank_B,
        offline_window=offline_window,
        offline_stride=offline_stride,
        lambda_A_off=lambda_A_off,
        lambda_B_off=lambda_B_off,
    )
    metadata = dict(basis.get("metadata") or {})
    metadata["source_path"] = str(baseline_path)
    metadata["source_mtime_ns"] = int(baseline_path.stat().st_mtime_ns)
    metadata["cache_path"] = str(cache_path.resolve())
    metadata["disturbance_profile"] = None if disturbance_profile is None else str(disturbance_profile).lower()
    metadata["cache_generated"] = True
    basis["metadata"] = metadata
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump(basis, handle)
    return basis


def build_or_load_polymer_reidentification_basis(
    *,
    baseline_path,
    cache_dir,
    A_ref: np.ndarray,
    B_ref: np.ndarray,
    rank_A: int,
    rank_B: int,
    offline_window: int,
    offline_stride: int,
    lambda_A_off: float,
    lambda_B_off: float,
    run_mode: str,
) -> dict:
    return build_or_load_reidentification_basis(
        baseline_path=baseline_path,
        cache_dir=cache_dir,
        A_ref=A_ref,
        B_ref=B_ref,
        basis_family=REIDENTIFICATION_BASIS_FAMILY,
        rank_A=rank_A,
        rank_B=rank_B,
        offline_window=offline_window,
        offline_stride=offline_stride,
        lambda_A_off=lambda_A_off,
        lambda_B_off=lambda_B_off,
        run_mode=run_mode,
    )


def assemble_batch_regression(batch: RollingIDBatch, A0_phys: np.ndarray, B0_phys: np.ndarray, basis: dict) -> tuple[np.ndarray, np.ndarray]:
    A0_phys = np.asarray(A0_phys, dtype=float)
    B0_phys = np.asarray(B0_phys, dtype=float)
    phi_blocks = []
    residual_blocks = []

    for x_t, u_t, x_tp1 in zip(batch.x_t, batch.u_t, batch.x_tp1):
        residual = x_tp1 - (A0_phys @ x_t) - (B0_phys @ u_t)
        feature_cols = [np.asarray(E, dtype=float) @ x_t for E in basis["A_basis"]]
        feature_cols.extend(np.asarray(F, dtype=float) @ u_t for F in basis["B_basis"])
        phi_blocks.append(np.column_stack(feature_cols))
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


def resolve_reidentification_theta_bounds(basis: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    theta_A_indices = np.asarray(basis["theta_A_indices"], dtype=int)
    theta_B_indices = np.asarray(basis["theta_B_indices"], dtype=int)
    theta_low = np.full(int(basis["theta_dim"]), -np.inf, dtype=float)
    theta_high = np.full(int(basis["theta_dim"]), np.inf, dtype=float)

    theta_low[theta_A_indices] = _expand_theta_vector(
        cfg["theta_low_A"],
        theta_A_indices.size,
        name="theta_low_A",
    )
    theta_high[theta_A_indices] = _expand_theta_vector(
        cfg["theta_high_A"],
        theta_A_indices.size,
        name="theta_high_A",
    )
    theta_low[theta_B_indices] = _expand_theta_vector(
        cfg["theta_low_B"],
        theta_B_indices.size,
        name="theta_low_B",
    )
    theta_high[theta_B_indices] = _expand_theta_vector(
        cfg["theta_high_B"],
        theta_B_indices.size,
        name="theta_high_B",
    )
    if np.any(theta_low > theta_high):
        raise ValueError("theta_low must be elementwise <= theta_high.")
    return theta_low.astype(float), theta_high.astype(float)


def resolve_reidentification_lambda_vectors(basis: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    theta_dim = int(basis["theta_dim"])
    theta_A_indices = np.asarray(basis["theta_A_indices"], dtype=int)
    theta_B_indices = np.asarray(basis["theta_B_indices"], dtype=int)

    lambda_prev_vec = np.full(theta_dim, float(cfg.get("lambda_prev", 1e-2)), dtype=float)
    lambda_0_vec = np.full(theta_dim, float(cfg.get("lambda_0", 1e-4)), dtype=float)
    lambda_prev_vec[theta_A_indices] = _expand_theta_vector(
        cfg["lambda_prev_A"],
        theta_A_indices.size,
        name="lambda_prev_A",
    )
    lambda_prev_vec[theta_B_indices] = _expand_theta_vector(
        cfg["lambda_prev_B"],
        theta_B_indices.size,
        name="lambda_prev_B",
    )
    lambda_0_vec[theta_A_indices] = _expand_theta_vector(
        cfg["lambda_0_A"],
        theta_A_indices.size,
        name="lambda_0_A",
    )
    lambda_0_vec[theta_B_indices] = _expand_theta_vector(
        cfg["lambda_0_B"],
        theta_B_indices.size,
        name="lambda_0_B",
    )
    return lambda_prev_vec.astype(float), lambda_0_vec.astype(float)


def _compute_regression_residual_norm(Phi: np.ndarray, residual_vec: np.ndarray, theta: np.ndarray) -> float:
    Phi = np.asarray(Phi, dtype=float)
    residual_vec = _as_1d_float(residual_vec)
    theta = _as_1d_float(theta, Phi.shape[1] if Phi.ndim == 2 else None)
    if Phi.ndim != 2:
        raise ValueError("Phi must be 2D.")
    return float(np.linalg.norm(residual_vec - Phi @ theta))


def _split_identification_batch(batch: RollingIDBatch, cfg: dict) -> tuple[RollingIDBatch, RollingIDBatch | None]:
    validation_fraction = float(cfg.get("guard_validation_fraction", REIDENTIFICATION_GUARD_VALIDATION_FRACTION))
    min_validation_samples = int(cfg.get("guard_min_validation_samples", REIDENTIFICATION_GUARD_MIN_VALIDATION_SAMPLES))
    min_train_samples = int(cfg.get("guard_min_train_samples", REIDENTIFICATION_GUARD_MIN_TRAIN_SAMPLES))

    n_samples = int(batch.x_t.shape[0])
    if validation_fraction <= 0.0 or n_samples <= 1:
        return batch, None

    min_train_samples = max(1, min_train_samples)
    min_validation_samples = max(1, min_validation_samples)
    if n_samples < min_train_samples + min_validation_samples:
        return batch, None

    proposed_validation = max(min_validation_samples, int(round(validation_fraction * n_samples)))
    validation_count = min(proposed_validation, n_samples - min_train_samples)
    if validation_count <= 0:
        return batch, None

    fit_batch = RollingIDBatch(
        x_t=np.asarray(batch.x_t[:-validation_count], dtype=float),
        u_t=np.asarray(batch.u_t[:-validation_count], dtype=float),
        x_tp1=np.asarray(batch.x_tp1[:-validation_count], dtype=float),
    )
    validation_batch = RollingIDBatch(
        x_t=np.asarray(batch.x_t[-validation_count:], dtype=float),
        u_t=np.asarray(batch.u_t[-validation_count:], dtype=float),
        x_tp1=np.asarray(batch.x_tp1[-validation_count:], dtype=float),
    )
    return fit_batch, validation_batch


def solve_batch_ridge(
    Phi: np.ndarray,
    residual_vec: np.ndarray,
    theta_prev: np.ndarray,
    lambda_prev_vec: np.ndarray,
    lambda_0_vec: np.ndarray,
    theta_low: np.ndarray,
    theta_high: np.ndarray,
) -> dict:
    Phi = np.asarray(Phi, dtype=float)
    residual_vec = _as_1d_float(residual_vec)
    theta_prev = _as_1d_float(theta_prev)
    theta_low = _as_1d_float(theta_low, theta_prev.size)
    theta_high = _as_1d_float(theta_high, theta_prev.size)
    lambda_prev_vec = _as_1d_float(lambda_prev_vec, theta_prev.size)
    lambda_0_vec = _as_1d_float(lambda_0_vec, theta_prev.size)

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
    lambda_prev_vec: np.ndarray,
    lambda_0_vec: np.ndarray,
    theta_low: np.ndarray,
    theta_high: np.ndarray,
) -> dict:
    Phi = np.asarray(Phi, dtype=float)
    residual_vec = _as_1d_float(residual_vec)
    theta_prev = _as_1d_float(theta_prev)
    theta_low = _as_1d_float(theta_low, theta_prev.size)
    theta_high = _as_1d_float(theta_high, theta_prev.size)
    lambda_prev_vec = _as_1d_float(lambda_prev_vec, theta_prev.size)
    lambda_0_vec = _as_1d_float(lambda_0_vec, theta_prev.size)

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


def solve_reidentification_batch(
    batch: RollingIDBatch,
    A0_phys: np.ndarray,
    B0_phys: np.ndarray,
    basis: dict,
    theta_prev: np.ndarray,
    cfg: dict,
) -> dict:
    fit_batch, validation_batch = _split_identification_batch(batch, cfg)
    Phi_fit, residual_fit = assemble_batch_regression(fit_batch, A0_phys=A0_phys, B0_phys=B0_phys, basis=basis)
    Phi_full, residual_full = assemble_batch_regression(batch, A0_phys=A0_phys, B0_phys=B0_phys, basis=basis)
    theta_prev = _as_1d_float(theta_prev, basis["theta_dim"])
    theta_low, theta_high = resolve_reidentification_theta_bounds(basis=basis, cfg=cfg)
    lambda_prev_vec, lambda_0_vec = resolve_reidentification_lambda_vectors(basis=basis, cfg=cfg)
    solver = str(cfg.get("id_solver", "ridge_closed_form")).lower()

    if solver == "ridge_closed_form":
        result = solve_batch_ridge(
            Phi=Phi_fit,
            residual_vec=residual_fit,
            theta_prev=theta_prev,
            lambda_prev_vec=lambda_prev_vec,
            lambda_0_vec=lambda_0_vec,
            theta_low=theta_low,
            theta_high=theta_high,
        )
    elif solver == "bounded_least_squares":
        result = solve_bounded_least_squares(
            Phi=Phi_fit,
            residual_vec=residual_fit,
            theta_prev=theta_prev,
            lambda_prev_vec=lambda_prev_vec,
            lambda_0_vec=lambda_0_vec,
            theta_low=theta_low,
            theta_high=theta_high,
        )
    else:
        raise ValueError("id_solver must be 'ridge_closed_form' or 'bounded_least_squares'.")

    candidate_theta = np.asarray(result["theta"], dtype=float)
    residual_norm_prev_train = _compute_regression_residual_norm(Phi_fit, residual_fit, theta_prev)
    residual_norm_candidate_train = float(result["residual_norm"])
    residual_norm_prev_full = _compute_regression_residual_norm(Phi_full, residual_full, theta_prev)
    residual_norm_candidate_full = _compute_regression_residual_norm(Phi_full, residual_full, candidate_theta)

    if validation_batch is not None:
        Phi_val, residual_val = assemble_batch_regression(validation_batch, A0_phys=A0_phys, B0_phys=B0_phys, basis=basis)
        residual_norm_prev_val = _compute_regression_residual_norm(Phi_val, residual_val, theta_prev)
        residual_norm_candidate_val = _compute_regression_residual_norm(Phi_val, residual_val, candidate_theta)
        validation_active = True
        validation_sample_count = int(validation_batch.x_t.shape[0])
    else:
        residual_norm_prev_val = float("nan")
        residual_norm_candidate_val = float("nan")
        validation_active = False
        validation_sample_count = 0

    eps = 1e-12
    result["residual_norm_train"] = residual_norm_candidate_train
    result["residual_norm"] = residual_norm_candidate_full
    result["residual_norm_prev_train"] = residual_norm_prev_train
    result["residual_norm_prev_full"] = residual_norm_prev_full
    result["residual_norm_val"] = residual_norm_candidate_val
    result["residual_norm_prev_val"] = residual_norm_prev_val
    result["residual_ratio_train"] = float(residual_norm_candidate_train / max(residual_norm_prev_train, eps))
    result["residual_ratio_full"] = float(residual_norm_candidate_full / max(residual_norm_prev_full, eps))
    result["residual_ratio_val"] = (
        float(residual_norm_candidate_val / max(residual_norm_prev_val, eps))
        if validation_active
        else float("nan")
    )
    result["Phi_shape"] = tuple(Phi_fit.shape)
    result["sample_count"] = int(batch.x_t.shape[0])
    result["train_sample_count"] = int(fit_batch.x_t.shape[0])
    result["validation_sample_count"] = validation_sample_count
    result["validation_active"] = validation_active
    result["theta_low"] = theta_low
    result["theta_high"] = theta_high
    result["lambda_prev_vec"] = lambda_prev_vec
    result["lambda_0_vec"] = lambda_0_vec
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
        "theta_within_bounds": bool(
            np.all(theta_candidate >= theta_low - tol) and np.all(theta_candidate <= theta_high + tol)
        ),
    }


def _guard_scale(value: float, soft: float, hard: float) -> float:
    value = float(value)
    soft = float(soft)
    hard = float(hard)
    if not np.isfinite(value):
        return 0.0
    if hard <= soft:
        return 1.0 if value <= soft else 0.0
    if value <= soft:
        return 1.0
    if value >= hard:
        return 0.0
    return float(1.0 - (value - soft) / (hard - soft))


def compute_blend_validity_scale(
    *,
    residual_norm: float,
    theta_clipped_fraction: float,
    condition_number: float,
    candidate_valid_flag: float,
    fallback_flag: float,
    identification_ready_flag: float,
    validity_mode: str = REIDENTIFICATION_BLEND_VALIDITY_MODE,
    scale_floor: float = REIDENTIFICATION_BLEND_VALIDITY_SCALE_FLOOR,
    residual_soft: float = REIDENTIFICATION_BLEND_VALIDITY_RESIDUAL_SOFT,
    residual_hard: float = REIDENTIFICATION_BLEND_VALIDITY_RESIDUAL_HARD,
    clipped_soft: float = REIDENTIFICATION_BLEND_VALIDITY_CLIPPED_SOFT,
    clipped_hard: float = REIDENTIFICATION_BLEND_VALIDITY_CLIPPED_HARD,
    condition_soft: float = REIDENTIFICATION_BLEND_VALIDITY_CONDITION_SOFT,
    condition_hard: float = REIDENTIFICATION_BLEND_VALIDITY_CONDITION_HARD,
    fallback_scale: float = REIDENTIFICATION_BLEND_VALIDITY_FALLBACK_SCALE,
    invalid_candidate_scale: float = REIDENTIFICATION_BLEND_VALIDITY_INVALID_CANDIDATE_SCALE,
) -> tuple[np.ndarray, dict]:
    mode = str(validity_mode).strip().lower()
    diagnostics = {
        "mode": mode,
        "identification_ready": bool(identification_ready_flag > 0.5),
        "residual_scale": 1.0,
        "clipped_scale": 1.0,
        "condition_scale": 1.0,
        "event_scale": 1.0,
    }
    if mode in {"", "none", "off"} or identification_ready_flag <= 0.5:
        return np.ones(2, dtype=float), diagnostics

    diagnostics["residual_scale"] = _guard_scale(residual_norm, residual_soft, residual_hard)
    diagnostics["clipped_scale"] = _guard_scale(theta_clipped_fraction, clipped_soft, clipped_hard)
    diagnostics["condition_scale"] = _guard_scale(condition_number, condition_soft, condition_hard)

    event_scale = 1.0
    if fallback_flag > 0.5:
        event_scale = min(event_scale, float(np.clip(fallback_scale, 0.0, 1.0)))
    if candidate_valid_flag <= 0.5:
        event_scale = min(event_scale, float(np.clip(invalid_candidate_scale, 0.0, 1.0)))
    diagnostics["event_scale"] = event_scale

    floor = float(np.clip(scale_floor, 0.0, 1.0))
    scalar = max(
        floor,
        min(
            diagnostics["residual_scale"],
            diagnostics["clipped_scale"],
            diagnostics["condition_scale"],
            diagnostics["event_scale"],
        ),
    )
    diagnostics["scalar_scale"] = scalar
    return np.full(2, scalar, dtype=float), diagnostics


def select_reidentified_model(
    *,
    A_candidate: np.ndarray | None,
    B_candidate: np.ndarray | None,
    theta_candidate: np.ndarray | None,
    theta_unclipped: np.ndarray | None,
    solve_success: bool,
    A0_phys: np.ndarray,
    B0_phys: np.ndarray,
    A_prev: np.ndarray,
    B_prev: np.ndarray,
    theta_prev: np.ndarray,
    theta_low: np.ndarray,
    theta_high: np.ndarray,
    delta_A_max: float,
    delta_B_max: float,
    guard_cfg: dict | None = None,
    solver_result: dict | None = None,
) -> dict:
    theta_low = _as_1d_float(theta_low, theta_prev.size)
    theta_high = _as_1d_float(theta_high, theta_prev.size)
    guard_cfg = {} if guard_cfg is None else dict(guard_cfg)
    guard_mode = str(guard_cfg.get("candidate_guard_mode", REIDENTIFICATION_CANDIDATE_GUARD_MODE)).strip().lower()

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
    theta_eval = compute_theta_clipping_diagnostics(
        theta_candidate=theta_candidate,
        theta_unclipped=theta_unclipped,
        theta_low=theta_low,
        theta_high=theta_high,
    )
    theta_eval["guard_mode"] = guard_mode

    eval_info = evaluate_identified_candidate(
        A_candidate=A_candidate,
        B_candidate=B_candidate,
        A0_phys=A0_phys,
        B0_phys=B0_phys,
        delta_A_max=delta_A_max,
        delta_B_max=delta_B_max,
    )
    eval_info["guard_mode"] = guard_mode

    guard_reason = str(eval_info["reason"])
    if eval_info["valid"] and guard_mode != "fro_only" and solver_result is not None:
        max_theta_clipped_fraction = float(
            guard_cfg.get("max_theta_clipped_fraction", REIDENTIFICATION_GUARD_MAX_THETA_CLIPPED_FRACTION)
        )
        max_condition_number = float(guard_cfg.get("max_condition_number", REIDENTIFICATION_GUARD_MAX_CONDITION_NUMBER))
        max_validation_residual_ratio = float(
            guard_cfg.get("max_validation_residual_ratio", REIDENTIFICATION_GUARD_MAX_VALIDATION_RESIDUAL_RATIO)
        )
        max_full_residual_ratio = float(
            guard_cfg.get("max_full_residual_ratio", REIDENTIFICATION_GUARD_MAX_FULL_RESIDUAL_RATIO)
        )
        validation_active = bool(solver_result.get("validation_active", False))
        validation_ratio = float(solver_result.get("residual_ratio_val", float("nan")))
        full_ratio = float(solver_result.get("residual_ratio_full", float("nan")))
        condition_number = float(solver_result.get("condition_number", float("nan")))
        clipped_fraction = float(theta_eval["clipped_fraction"])

        eval_info["validation_active"] = validation_active
        eval_info["validation_residual_ratio"] = validation_ratio
        eval_info["full_residual_ratio"] = full_ratio
        eval_info["condition_number"] = condition_number
        eval_info["max_theta_clipped_fraction"] = max_theta_clipped_fraction
        eval_info["max_condition_number"] = max_condition_number
        eval_info["max_validation_residual_ratio"] = max_validation_residual_ratio
        eval_info["max_full_residual_ratio"] = max_full_residual_ratio

        if clipped_fraction > max_theta_clipped_fraction:
            eval_info["valid"] = False
            guard_reason = "theta_clipping_guard"
        elif not np.isfinite(condition_number) or condition_number > max_condition_number:
            eval_info["valid"] = False
            guard_reason = "condition_guard"
        elif validation_active and np.isfinite(validation_ratio) and validation_ratio > max_validation_residual_ratio:
            eval_info["valid"] = False
            guard_reason = "validation_residual_guard"
        elif np.isfinite(full_ratio) and full_ratio > max_full_residual_ratio:
            eval_info["valid"] = False
            guard_reason = "full_residual_guard"

        eval_info["reason"] = guard_reason
    theta_eval["guard_reason"] = guard_reason

    if eval_info["valid"]:
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


def map_action_to_dual_eta(raw_action) -> tuple[np.ndarray, np.ndarray]:
    raw_action = np.asarray(raw_action, dtype=float).reshape(-1)
    if raw_action.size == 1:
        raw_action = np.repeat(raw_action, 2)
    if raw_action.size != 2:
        raise ValueError("Dual-eta action must contain exactly 2 entries.")
    raw_action = np.clip(raw_action, -1.0, 1.0)
    eta_raw = np.clip(0.5 * (raw_action + 1.0), 0.0, 1.0)
    return raw_action.astype(float), eta_raw.astype(float)


def eta_to_raw_action(eta) -> np.ndarray:
    eta = np.asarray(eta, dtype=float).reshape(-1)
    if eta.size == 1:
        eta = np.repeat(eta, 2)
    if eta.size != 2:
        raise ValueError("eta must be scalar or length 2 for the dual-eta policy.")
    eta = np.clip(eta, 0.0, 1.0)
    return np.clip(2.0 * eta - 1.0, -1.0, 1.0).astype(float)


def smooth_eta(eta_prev, eta_raw, tau_eta) -> np.ndarray:
    eta_prev = np.asarray(eta_prev, dtype=float).reshape(-1)
    eta_raw = np.asarray(eta_raw, dtype=float).reshape(-1)
    tau_eta = np.asarray(tau_eta, dtype=float).reshape(-1)
    if eta_prev.size == 1:
        eta_prev = np.repeat(eta_prev, 2)
    if eta_raw.size == 1:
        eta_raw = np.repeat(eta_raw, 2)
    if tau_eta.size == 1:
        tau_eta = np.repeat(tau_eta, 2)
    if eta_prev.size != 2 or eta_raw.size != 2 or tau_eta.size != 2:
        raise ValueError("Dual-eta smoothing expects 2-element vectors.")
    tau_eta = np.clip(tau_eta, 0.0, 1.0)
    return ((1.0 - tau_eta) * eta_prev + tau_eta * eta_raw).astype(float)


def normalize_force_eta_constant(force_eta_constant) -> np.ndarray | None:
    if force_eta_constant is None:
        return None
    return np.clip(np.asarray(eta_to_raw_action(force_eta_constant), dtype=float) * 0.5 + 0.5, 0.0, 1.0)


def blend_prediction_model(
    A0_phys: np.ndarray,
    B0_phys: np.ndarray,
    A_id_phys: np.ndarray,
    B_id_phys: np.ndarray,
    *,
    eta_A: float,
    eta_B: float,
) -> tuple[np.ndarray, np.ndarray]:
    eta_A = float(np.clip(eta_A, 0.0, 1.0))
    eta_B = float(np.clip(eta_B, 0.0, 1.0))
    A0_phys = np.asarray(A0_phys, dtype=float)
    B0_phys = np.asarray(B0_phys, dtype=float)
    A_id_phys = np.asarray(A_id_phys, dtype=float)
    B_id_phys = np.asarray(B_id_phys, dtype=float)
    A_pred = A0_phys + eta_A * (A_id_phys - A0_phys)
    B_pred = B0_phys + eta_B * (B_id_phys - B0_phys)
    return A_pred, B_pred


def build_reidentification_policy_state(
    base_state,
    *,
    prev_eta_A: float,
    prev_eta_B: float,
    residual_norm: float,
    active_A_ratio: float,
    active_B_ratio: float,
    candidate_valid_flag: float,
    observer_refresh_success_flag: float,
    fallback_flag: float,
    delta_A_max: float,
    delta_B_max: float,
    blend_extra_clip: float = REIDENTIFICATION_BLEND_EXTRA_CLIP,
    blend_residual_scale: float = REIDENTIFICATION_BLEND_RESIDUAL_SCALE,
) -> np.ndarray:
    base_state = _as_1d_float(base_state)
    eps = 1e-8
    z_eta_A = float(np.clip(2.0 * float(np.clip(prev_eta_A, 0.0, 1.0)) - 1.0, -1.0, 1.0))
    z_eta_B = float(np.clip(2.0 * float(np.clip(prev_eta_B, 0.0, 1.0)) - 1.0, -1.0, 1.0))
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
    extras = np.asarray(
        [
            z_eta_A,
            z_eta_B,
            z_r,
            z_A,
            z_B,
            float(candidate_valid_flag),
            float(observer_refresh_success_flag),
            float(fallback_flag),
        ],
        dtype=np.float32,
    )
    return np.concatenate((base_state.astype(np.float32), extras), axis=0).astype(np.float32)


def get_reidentification_state_dim(base_aug_dim: int, n_outputs: int, n_inputs: int, state_mode: str) -> int:
    return int(get_rl_state_dim(base_aug_dim, n_outputs, n_inputs, state_mode)) + 8


def build_observer_refresh_candidate(
    *,
    A_obs_current_phys: np.ndarray,
    B_obs_current_phys: np.ndarray,
    A_interval_mean: np.ndarray,
    B_interval_mean: np.ndarray,
    rho_obs: float,
) -> tuple[np.ndarray, np.ndarray]:
    rho_obs = float(np.clip(float(rho_obs), 0.0, 1.0))
    A_obs_current_phys = np.asarray(A_obs_current_phys, dtype=float)
    B_obs_current_phys = np.asarray(B_obs_current_phys, dtype=float)
    A_interval_mean = np.asarray(A_interval_mean, dtype=float)
    B_interval_mean = np.asarray(B_interval_mean, dtype=float)
    A_obs_cand = (1.0 - rho_obs) * A_obs_current_phys + rho_obs * A_interval_mean
    B_obs_cand = (1.0 - rho_obs) * B_obs_current_phys + rho_obs * B_interval_mean
    return A_obs_cand.astype(float), B_obs_cand.astype(float)


def attempt_observer_refresh(
    *,
    A_obs_current_phys: np.ndarray,
    B_obs_current_phys: np.ndarray,
    A_interval_mean: np.ndarray,
    B_interval_mean: np.ndarray,
    A0_phys: np.ndarray,
    B0_phys: np.ndarray,
    C_phys: np.ndarray,
    C_aug: np.ndarray,
    poles: np.ndarray,
    rho_obs: float,
    delta_A_max: float,
    delta_B_max: float,
) -> dict:
    A_obs_cand, B_obs_cand = build_observer_refresh_candidate(
        A_obs_current_phys=A_obs_current_phys,
        B_obs_current_phys=B_obs_current_phys,
        A_interval_mean=A_interval_mean,
        B_interval_mean=B_interval_mean,
        rho_obs=rho_obs,
    )
    eval_info = evaluate_identified_candidate(
        A_candidate=A_obs_cand,
        B_candidate=B_obs_cand,
        A0_phys=A0_phys,
        B0_phys=B0_phys,
        delta_A_max=delta_A_max,
        delta_B_max=delta_B_max,
    )
    if not eval_info["valid"]:
        return {
            "accepted": False,
            "reason": f"observer_{eval_info['reason']}",
            "A_obs_phys": np.asarray(A_obs_current_phys, dtype=float),
            "B_obs_phys": np.asarray(B_obs_current_phys, dtype=float),
            "A_obs_ratio": float(_relative_fro(A_obs_current_phys, A0_phys)),
            "B_obs_ratio": float(_relative_fro(B_obs_current_phys, B0_phys)),
        }

    try:
        A_obs_aug, B_obs_aug, _ = augment_state_space(A_obs_cand, B_obs_cand, C_phys)
        L_obs_new = compute_observer_gain(A_obs_aug, C_aug, poles)
    except Exception as exc:
        return {
            "accepted": False,
            "reason": f"observer_gain_failure:{exc}",
            "A_obs_phys": np.asarray(A_obs_current_phys, dtype=float),
            "B_obs_phys": np.asarray(B_obs_current_phys, dtype=float),
            "A_obs_ratio": float(_relative_fro(A_obs_current_phys, A0_phys)),
            "B_obs_ratio": float(_relative_fro(B_obs_current_phys, B0_phys)),
        }

    if not (np.all(np.isfinite(A_obs_aug)) and np.all(np.isfinite(B_obs_aug)) and np.all(np.isfinite(L_obs_new))):
        return {
            "accepted": False,
            "reason": "observer_nonfinite",
            "A_obs_phys": np.asarray(A_obs_current_phys, dtype=float),
            "B_obs_phys": np.asarray(B_obs_current_phys, dtype=float),
            "A_obs_ratio": float(_relative_fro(A_obs_current_phys, A0_phys)),
            "B_obs_ratio": float(_relative_fro(B_obs_current_phys, B0_phys)),
        }

    return {
        "accepted": True,
        "reason": "accepted",
        "A_obs_phys": np.asarray(A_obs_cand, dtype=float),
        "B_obs_phys": np.asarray(B_obs_cand, dtype=float),
        "A_obs_aug": np.asarray(A_obs_aug, dtype=float),
        "B_obs_aug": np.asarray(B_obs_aug, dtype=float),
        "L_obs": np.asarray(L_obs_new, dtype=float),
        "A_obs_ratio": float(_relative_fro(A_obs_cand, A0_phys)),
        "B_obs_ratio": float(_relative_fro(B_obs_cand, B0_phys)),
    }


__all__ = [
    "REIDENTIFICATION_BASIS_FAMILY",
    "REIDENTIFICATION_COMPONENT_MODE",
    "REIDENTIFICATION_OBSERVER_ALIGNMENT",
    "REIDENTIFICATION_CANDIDATE_GUARD_MODE",
    "REIDENTIFICATION_NORMALIZE_BLEND_EXTRAS",
    "REIDENTIFICATION_BLEND_EXTRA_CLIP",
    "REIDENTIFICATION_BLEND_RESIDUAL_SCALE",
    "REIDENTIFICATION_LOG_THETA_CLIPPING",
    "REIDENTIFICATION_GUARD_VALIDATION_FRACTION",
    "REIDENTIFICATION_GUARD_MIN_VALIDATION_SAMPLES",
    "REIDENTIFICATION_GUARD_MIN_TRAIN_SAMPLES",
    "REIDENTIFICATION_GUARD_MAX_THETA_CLIPPED_FRACTION",
    "REIDENTIFICATION_GUARD_MAX_CONDITION_NUMBER",
    "REIDENTIFICATION_GUARD_MAX_VALIDATION_RESIDUAL_RATIO",
    "REIDENTIFICATION_GUARD_MAX_FULL_RESIDUAL_RATIO",
    "REIDENTIFICATION_BLEND_VALIDITY_MODE",
    "REIDENTIFICATION_BLEND_VALIDITY_SCALE_FLOOR",
    "REIDENTIFICATION_BLEND_VALIDITY_RESIDUAL_SOFT",
    "REIDENTIFICATION_BLEND_VALIDITY_RESIDUAL_HARD",
    "REIDENTIFICATION_BLEND_VALIDITY_CLIPPED_SOFT",
    "REIDENTIFICATION_BLEND_VALIDITY_CLIPPED_HARD",
    "REIDENTIFICATION_BLEND_VALIDITY_CONDITION_SOFT",
    "REIDENTIFICATION_BLEND_VALIDITY_CONDITION_HARD",
    "REIDENTIFICATION_BLEND_VALIDITY_FALLBACK_SCALE",
    "REIDENTIFICATION_BLEND_VALIDITY_INVALID_CANDIDATE_SCALE",
    "RollingIDBatch",
    "RollingIDBuffer",
    "assemble_batch_regression",
    "attempt_observer_refresh",
    "blend_prediction_model",
    "build_observer_refresh_candidate",
    "build_or_load_reidentification_basis",
    "build_or_load_polymer_reidentification_basis",
    "build_reidentification_policy_state",
    "compute_blend_validity_scale",
    "compute_theta_clipping_diagnostics",
    "eta_to_raw_action",
    "evaluate_identified_candidate",
    "extract_lowrank_residual_basis_from_baseline",
    "get_reidentification_state_dim",
    "map_action_to_dual_eta",
    "normalize_force_eta_constant",
    "reconstruct_model_from_theta",
    "resolve_basis_family",
    "resolve_lowrank_basis_cache_path",
    "resolve_reidentification_lambda_vectors",
    "resolve_reidentification_theta_bounds",
    "select_reidentified_model",
    "smooth_eta",
    "solve_batch_ridge",
    "solve_bounded_least_squares",
    "solve_dense_local_residual_fit",
    "solve_reidentification_batch",
]
