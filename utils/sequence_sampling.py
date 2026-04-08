from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceIndexBatch:
    start_positions: np.ndarray
    physical_start_indices: np.ndarray
    index_matrix: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    is_weights: np.ndarray


def ordered_ring_indices(size: int, ptr: int, capacity: int) -> np.ndarray:
    size = int(size)
    ptr = int(ptr)
    capacity = int(capacity)
    if size <= 0:
        return np.zeros((0,), dtype=np.int64)
    if size < capacity:
        return np.arange(size, dtype=np.int64)
    return np.concatenate(
        [
            np.arange(ptr, capacity, dtype=np.int64),
            np.arange(0, ptr, dtype=np.int64),
        ]
    )


def sample_hybrid_start_positions(
    *,
    size: int,
    batch_size: int,
    recent_window: int,
    recent_frac: float,
    per_frac: float,
    priorities_ordered: np.ndarray | None = None,
    alpha: float = 0.6,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    size = int(size)
    batch_size = int(batch_size)
    if size <= 0:
        raise ValueError("Cannot sample sequences from an empty buffer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    k_per = int(batch_size * float(per_frac)) if priorities_ordered is not None else 0
    k_recent = int(batch_size * float(recent_frac))
    k_uniform = batch_size - k_per - k_recent

    all_positions = np.arange(size, dtype=np.int64)
    recent_window = max(1, min(size, int(recent_window)))
    recent_positions = all_positions[-recent_window:]

    recent_replace = recent_positions.size < k_recent
    idx_recent = (
        np.random.choice(recent_positions, size=k_recent, replace=recent_replace)
        if k_recent > 0
        else np.zeros((0,), dtype=np.int64)
    )
    idx_uniform = (
        np.random.choice(all_positions, size=k_uniform, replace=True)
        if k_uniform > 0
        else np.zeros((0,), dtype=np.int64)
    )

    if k_per > 0:
        priorities_ordered = np.asarray(priorities_ordered, dtype=np.float64).reshape(-1)
        if priorities_ordered.size != size:
            raise ValueError("priorities_ordered must match the ordered buffer size.")
        probs = np.maximum(priorities_ordered, float(eps)) ** float(alpha)
        probs_sum = probs.sum()
        if not np.isfinite(probs_sum) or probs_sum <= 0.0:
            probs = np.full(size, 1.0 / size, dtype=np.float64)
        else:
            probs /= probs_sum
        idx_per = np.random.choice(all_positions, size=k_per, replace=True, p=probs)
        weights = (size * probs[idx_per]) ** (-float(beta))
        weights /= np.maximum(weights.max(), 1e-12)
        is_w_per = weights.astype(np.float32)
    else:
        idx_per = np.zeros((0,), dtype=np.int64)
        is_w_per = np.zeros((0,), dtype=np.float32)

    start_positions = np.concatenate([idx_per, idx_recent, idx_uniform], axis=0)
    is_weights = np.concatenate(
        [is_w_per, np.ones(k_recent + k_uniform, dtype=np.float32)],
        axis=0,
    )
    return start_positions.astype(np.int64), is_weights.astype(np.float32)


def build_sequence_index_batch(
    *,
    ordered_indices: np.ndarray,
    episode_ids: np.ndarray,
    dones: np.ndarray,
    start_positions: np.ndarray,
    seq_len: int,
    is_weights: np.ndarray | None = None,
) -> SequenceIndexBatch:
    ordered_indices = np.asarray(ordered_indices, dtype=np.int64).reshape(-1)
    start_positions = np.asarray(start_positions, dtype=np.int64).reshape(-1)
    episode_ids = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    seq_len = int(seq_len)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if ordered_indices.size == 0:
        raise ValueError("ordered_indices must not be empty.")

    n_batch = start_positions.size
    index_matrix = np.zeros((n_batch, seq_len), dtype=np.int64)
    mask = np.zeros((n_batch, seq_len), dtype=np.float32)
    lengths = np.zeros((n_batch,), dtype=np.int32)
    physical_start_indices = np.zeros((n_batch,), dtype=np.int64)

    for row, start in enumerate(start_positions):
        start = int(np.clip(start, 0, ordered_indices.size - 1))
        start_phys = ordered_indices[start]
        physical_start_indices[row] = start_phys
        episode = int(episode_ids[start_phys])
        fill_phys = start_phys
        for step in range(seq_len):
            logical_pos = start + step
            if logical_pos >= ordered_indices.size:
                break
            phys_idx = int(ordered_indices[logical_pos])
            if int(episode_ids[phys_idx]) != episode:
                break
            fill_phys = phys_idx
            index_matrix[row, step] = phys_idx
            mask[row, step] = 1.0
            lengths[row] += 1
            if bool(dones[phys_idx]):
                break
        if lengths[row] == 0:
            index_matrix[row, 0] = fill_phys

    if is_weights is None:
        is_weights = np.ones((n_batch,), dtype=np.float32)
    else:
        is_weights = np.asarray(is_weights, dtype=np.float32).reshape(-1)
        if is_weights.size != n_batch:
            raise ValueError("is_weights must match the number of sampled sequences.")

    return SequenceIndexBatch(
        start_positions=start_positions,
        physical_start_indices=physical_start_indices,
        index_matrix=index_matrix,
        mask=mask,
        lengths=lengths,
        is_weights=is_weights,
    )
