import numpy as np

from utils.multiplier_mapping import map_centered_action_to_bounds, map_centered_bounds_to_action


RANGE_PROFILES = {
    "tight": {
        "diag_low": 0.98,
        "diag_high": 1.02,
        "off_low": 0.95,
        "off_high": 1.05,
        "b_low": 0.97,
        "b_high": 1.03,
    },
    "default": {
        "diag_low": 0.97,
        "diag_high": 1.03,
        "off_low": 0.93,
        "off_high": 1.07,
        "b_low": 0.95,
        "b_high": 1.05,
    },
    "wide": {
        "diag_low": 0.75,
        "diag_high": 1.25,
        "off_low": 0.75,
        "off_high": 1.25,
        "b_low": 0.75,
        "b_high": 1.25,
    },
}


def _as_float_array(value, name):
    arr = np.asarray(value, float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return arr


def _require_finite(name, arr):
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must not contain infs or NaNs.")


def validate_positive_bounds(low, high):
    low = _as_float_array(low, "low")
    high = _as_float_array(high, "high")
    if low.shape != high.shape:
        raise ValueError("low/high bounds must have the same shape.")
    _require_finite("low", low)
    _require_finite("high", high)
    if np.any(low <= 0.0):
        raise ValueError("All multiplier lower bounds must be strictly positive.")
    if np.any(high <= low):
        raise ValueError("All multiplier upper bounds must be greater than the lower bounds.")
    return low, high


def _broadcast_positive_override(base, override, name):
    if override is None:
        return np.asarray(base, float)
    arr = np.asarray(override, float)
    if arr.ndim == 0:
        out = np.full_like(np.asarray(base, float), float(arr))
    elif arr.shape == np.asarray(base, float).shape:
        out = arr.astype(float, copy=True)
    else:
        raise ValueError(f"{name} override must be a scalar or match shape {np.asarray(base, float).shape}.")
    return out


def split_augmented_model(A_aug, B_aug, n_outputs, atol=1e-10):
    A_aug = np.asarray(A_aug, float)
    B_aug = np.asarray(B_aug, float)
    if A_aug.ndim != 2 or A_aug.shape[0] != A_aug.shape[1]:
        raise ValueError("A_aug must be a square 2D array.")
    if B_aug.ndim != 2 or B_aug.shape[0] != A_aug.shape[0]:
        raise ValueError("B_aug must be 2D and share the row dimension of A_aug.")
    n_outputs = int(n_outputs)
    if n_outputs <= 0 or n_outputs >= A_aug.shape[0]:
        raise ValueError("n_outputs must define a non-empty disturbance block.")

    n_states = int(A_aug.shape[0])
    n_inputs = int(B_aug.shape[1])
    n_phys = n_states - n_outputs

    A_phys = A_aug[:n_phys, :n_phys].copy()
    B_phys = B_aug[:n_phys, :].copy()
    upper_right = A_aug[:n_phys, n_phys:].copy()
    lower_left = A_aug[n_phys:, :n_phys].copy()
    lower_right = A_aug[n_phys:, n_phys:].copy()
    B_disturbance = B_aug[n_phys:, :].copy()

    if not np.allclose(upper_right, 0.0, atol=atol):
        raise ValueError("Augmented model upper-right disturbance coupling block must remain zero.")
    if not np.allclose(lower_left, 0.0, atol=atol):
        raise ValueError("Augmented model lower-left block must remain zero.")
    if not np.allclose(lower_right, np.eye(n_outputs), atol=atol):
        raise ValueError("Augmented disturbance block must be the identity matrix.")
    if not np.allclose(B_disturbance, 0.0, atol=atol):
        raise ValueError("Augmented disturbance input block must remain zero.")

    return {
        "A_aug": A_aug.copy(),
        "B_aug": B_aug.copy(),
        "A_phys": A_phys,
        "B_phys": B_phys,
        "upper_right": upper_right,
        "lower_left": lower_left,
        "lower_right": lower_right,
        "B_disturbance": B_disturbance,
        "n_states": n_states,
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
        "n_phys": n_phys,
    }


def map_normalized_action_to_multipliers(action, low, high):
    low, high = validate_positive_bounds(low, high)
    return map_centered_action_to_bounds(action, low, high, nominal=1.0)


def map_multipliers_to_normalized_action(value, low, high):
    low, high = validate_positive_bounds(low, high)
    return map_centered_bounds_to_action(value, low, high, nominal=1.0)


def resolve_block_groups(n_phys, group_count=None, groups=None):
    n_phys = int(n_phys)
    if n_phys <= 0:
        raise ValueError("n_phys must be positive.")

    if groups is None:
        if group_count is None:
            raise ValueError("Either group_count or groups must be provided.")
        group_count = int(group_count)
        if group_count <= 0 or group_count > n_phys:
            raise ValueError("group_count must be between 1 and n_phys.")
        groups = [chunk.tolist() for chunk in np.array_split(np.arange(n_phys, dtype=int), group_count)]
    else:
        groups = [list(map(int, grp)) for grp in groups]
        if len(groups) == 0:
            raise ValueError("groups must not be empty.")

    seen = []
    resolved = []
    for grp in groups:
        if len(grp) == 0:
            raise ValueError("Block groups must not contain empty groups.")
        if any(idx < 0 or idx >= n_phys for idx in grp):
            raise ValueError("Block-group indices must lie inside the physical-state range.")
        resolved.append(tuple(grp))
        seen.extend(grp)

    if sorted(seen) != list(range(n_phys)):
        raise ValueError("Block groups must partition the physical states exactly once.")

    return tuple(resolved)


def resolve_band_offsets(offsets, n_phys):
    if offsets is None:
        offsets = [0, 1, 2]
    resolved = sorted({int(v) for v in offsets})
    if len(resolved) == 0:
        raise ValueError("band offsets must not be empty.")
    if resolved[0] != 0:
        raise ValueError("band offsets must include 0 for the main band.")
    if any(v < 0 for v in resolved):
        raise ValueError("band offsets must be non-negative.")
    if any(v >= int(n_phys) for v in resolved):
        raise ValueError("band offsets must be smaller than the number of physical states.")
    return tuple(resolved)


def resolve_range_profile(range_profile):
    key = str(range_profile).strip().lower()
    if key not in RANGE_PROFILES:
        raise KeyError(f"Unknown structured range profile: {range_profile}")
    return key, RANGE_PROFILES[key].copy()


def build_structured_update_spec(
    A_aug,
    B_aug,
    n_outputs,
    update_family,
    range_profile="tight",
    block_group_count=3,
    block_groups=None,
    band_offsets=None,
    a_low_override=None,
    a_high_override=None,
    b_low_override=None,
    b_high_override=None,
):
    blocks = split_augmented_model(A_aug, B_aug, n_outputs)
    update_family = str(update_family).strip().lower()
    if update_family not in {"block", "band"}:
        raise ValueError("update_family must be 'block' or 'band'.")

    profile_name, profile = resolve_range_profile(range_profile)
    n_inputs = int(blocks["n_inputs"])
    n_phys = int(blocks["n_phys"])

    if update_family == "block":
        groups = resolve_block_groups(n_phys, group_count=block_group_count, groups=block_groups)
        a_labels = [f"A_block_{idx + 1}" for idx in range(len(groups))] + ["A_off"]
        low_a = np.array([profile["diag_low"]] * len(groups) + [profile["off_low"]], float)
        high_a = np.array([profile["diag_high"]] * len(groups) + [profile["off_high"]], float)
        band_meta = None
        block_meta = {
            "group_count": len(groups),
            "groups": tuple(tuple(int(v) for v in grp) for grp in groups),
        }
    else:
        offsets = resolve_band_offsets(band_offsets, n_phys=n_phys)
        a_labels = [f"A_band_{offset}" for offset in offsets]
        low_a = np.array(
            [profile["diag_low"] if offset == 0 else profile["off_low"] for offset in offsets],
            float,
        )
        high_a = np.array(
            [profile["diag_high"] if offset == 0 else profile["off_high"] for offset in offsets],
            float,
        )
        block_meta = None
        band_meta = {"offsets": tuple(int(v) for v in offsets)}

    b_labels = [f"B_col_{idx + 1}" for idx in range(n_inputs)]
    low_b = np.full(n_inputs, profile["b_low"], float)
    high_b = np.full(n_inputs, profile["b_high"], float)

    low_a = _broadcast_positive_override(low_a, a_low_override, "a_low")
    high_a = _broadcast_positive_override(high_a, a_high_override, "a_high")
    low_b = _broadcast_positive_override(low_b, b_low_override, "b_low")
    high_b = _broadcast_positive_override(high_b, b_high_override, "b_high")

    low_bounds = np.concatenate([low_a, low_b])
    high_bounds = np.concatenate([high_a, high_b])
    validate_positive_bounds(low_bounds, high_bounds)

    return {
        "update_family": update_family,
        "range_profile": profile_name,
        "range_profile_values": profile,
        "n_phys": n_phys,
        "n_inputs": n_inputs,
        "n_outputs": int(blocks["n_outputs"]),
        "action_dim": int(low_bounds.size),
        "a_dim": int(low_a.size),
        "b_dim": int(low_b.size),
        "low_bounds": low_bounds,
        "high_bounds": high_bounds,
        "low_a": low_a,
        "high_a": high_a,
        "low_b": low_b,
        "high_b": high_b,
        "action_labels": tuple(a_labels + b_labels),
        "theta_a_labels": tuple(a_labels),
        "theta_b_labels": tuple(b_labels),
        "block_cfg": block_meta,
        "band_cfg": band_meta,
    }


def validate_preserved_augmented_structure(base_split, A_aug_new, B_aug_new, atol=1e-10):
    A_aug_new = np.asarray(A_aug_new, float)
    B_aug_new = np.asarray(B_aug_new, float)
    n_phys = int(base_split["n_phys"])
    n_outputs = int(base_split["n_outputs"])

    if A_aug_new.shape != base_split["A_aug"].shape:
        raise ValueError("Updated A_aug shape does not match the nominal augmented model.")
    if B_aug_new.shape != base_split["B_aug"].shape:
        raise ValueError("Updated B_aug shape does not match the nominal augmented model.")
    _require_finite("A_aug_new", A_aug_new)
    _require_finite("B_aug_new", B_aug_new)

    if not np.allclose(A_aug_new[:n_phys, n_phys:], base_split["upper_right"], atol=atol):
        raise ValueError("Updated A_aug must preserve the upper-right zero disturbance block.")
    if not np.allclose(A_aug_new[n_phys:, :n_phys], base_split["lower_left"], atol=atol):
        raise ValueError("Updated A_aug must preserve the lower-left zero block.")
    if not np.allclose(A_aug_new[n_phys:, n_phys:], np.eye(n_outputs), atol=atol):
        raise ValueError("Updated A_aug must preserve the disturbance identity block.")
    if not np.allclose(B_aug_new[n_phys:, :], base_split["B_disturbance"], atol=atol):
        raise ValueError("Updated B_aug must preserve the disturbance-input zero block.")


def _frob_ratio(updated, nominal):
    denom = float(np.linalg.norm(nominal, ord="fro"))
    if denom <= 0.0:
        return 0.0
    return float(np.linalg.norm(updated - nominal, ord="fro") / denom)


def _physical_spectral_radius(A_phys):
    eigvals = np.linalg.eigvals(A_phys)
    if eigvals.size == 0:
        return 0.0
    return float(np.max(np.abs(eigvals)))


def build_block_scaled_model(A_aug, B_aug, n_outputs, block_cfg, theta_A, theta_B):
    base = split_augmented_model(A_aug, B_aug, n_outputs)
    groups = resolve_block_groups(
        base["n_phys"],
        group_count=(block_cfg or {}).get("group_count"),
        groups=(block_cfg or {}).get("groups"),
    )
    theta_A = _as_float_array(theta_A, "theta_A")
    theta_B = _as_float_array(theta_B, "theta_B")
    if theta_A.size != len(groups) + 1:
        raise ValueError("Block-lite A multipliers must have length group_count + 1.")
    if theta_B.size != base["n_inputs"]:
        raise ValueError("B multipliers must have one value per manipulated input.")
    if np.any(theta_A <= 0.0) or np.any(theta_B <= 0.0):
        raise ValueError("All structured multipliers must remain strictly positive.")

    A_phys = base["A_phys"].copy()
    B_phys = base["B_phys"].copy()

    diag_multipliers = theta_A[:-1]
    off_multiplier = float(theta_A[-1])

    for row_idx, row_group in enumerate(groups):
        row_ix = np.ix_(row_group, row_group)
        A_phys[row_ix] *= float(diag_multipliers[row_idx])
        for col_idx, col_group in enumerate(groups):
            if row_idx == col_idx:
                continue
            A_phys[np.ix_(row_group, col_group)] *= off_multiplier

    B_phys *= theta_B.reshape(1, -1)

    A_aug_new = base["A_aug"].copy()
    B_aug_new = base["B_aug"].copy()
    A_aug_new[: base["n_phys"], : base["n_phys"]] = A_phys
    B_aug_new[: base["n_phys"], :] = B_phys

    validate_preserved_augmented_structure(base, A_aug_new, B_aug_new)
    return {
        "A_aug": A_aug_new,
        "B_aug": B_aug_new,
        "A_phys": A_phys,
        "B_phys": B_phys,
        "theta_a": theta_A.copy(),
        "theta_b": theta_B.copy(),
        "block_groups": tuple(tuple(int(v) for v in grp) for grp in groups),
        "A_fro_ratio": _frob_ratio(A_phys, base["A_phys"]),
        "B_fro_ratio": _frob_ratio(B_phys, base["B_phys"]),
        "spectral_radius": _physical_spectral_radius(A_phys),
    }


def build_band_scaled_model(A_aug, B_aug, n_outputs, band_cfg, theta_A, theta_B):
    base = split_augmented_model(A_aug, B_aug, n_outputs)
    offsets = resolve_band_offsets((band_cfg or {}).get("offsets"), n_phys=base["n_phys"])
    theta_A = _as_float_array(theta_A, "theta_A")
    theta_B = _as_float_array(theta_B, "theta_B")
    if theta_A.size != len(offsets):
        raise ValueError("Band-lite A multipliers must have one value per configured offset.")
    if theta_B.size != base["n_inputs"]:
        raise ValueError("B multipliers must have one value per manipulated input.")
    if np.any(theta_A <= 0.0) or np.any(theta_B <= 0.0):
        raise ValueError("All structured multipliers must remain strictly positive.")

    A_phys = base["A_phys"].copy()
    B_phys = base["B_phys"].copy()
    offset_to_multiplier = {int(offset): float(theta_A[idx]) for idx, offset in enumerate(offsets)}
    n_phys = int(base["n_phys"])

    for i in range(n_phys):
        for j in range(n_phys):
            offset = abs(i - j)
            if offset in offset_to_multiplier:
                A_phys[i, j] *= offset_to_multiplier[offset]

    B_phys *= theta_B.reshape(1, -1)

    A_aug_new = base["A_aug"].copy()
    B_aug_new = base["B_aug"].copy()
    A_aug_new[:n_phys, :n_phys] = A_phys
    B_aug_new[:n_phys, :] = B_phys

    validate_preserved_augmented_structure(base, A_aug_new, B_aug_new)
    return {
        "A_aug": A_aug_new,
        "B_aug": B_aug_new,
        "A_phys": A_phys,
        "B_phys": B_phys,
        "theta_a": theta_A.copy(),
        "theta_b": theta_B.copy(),
        "band_offsets": tuple(int(v) for v in offsets),
        "A_fro_ratio": _frob_ratio(A_phys, base["A_phys"]),
        "B_fro_ratio": _frob_ratio(B_phys, base["B_phys"]),
        "spectral_radius": _physical_spectral_radius(A_phys),
    }


__all__ = [
    "RANGE_PROFILES",
    "build_band_scaled_model",
    "build_block_scaled_model",
    "build_structured_update_spec",
    "map_multipliers_to_normalized_action",
    "map_normalized_action_to_multipliers",
    "resolve_band_offsets",
    "resolve_block_groups",
    "resolve_range_profile",
    "split_augmented_model",
    "validate_positive_bounds",
    "validate_preserved_augmented_structure",
]
