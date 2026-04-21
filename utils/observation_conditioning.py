import numpy as np


VALID_BASE_STATE_NORM_MODES = {"fixed_minmax", "running_zscore_physical_xhat"}
VALID_MISMATCH_TRANSFORM_MODES = {"hard_clip", "soft_tanh", "signed_log"}
VALID_OBSERVER_ALIGNMENT_MODES = {"legacy_previous_measurement", "current_measurement_corrector"}


class RunningFeatureNormalizer:
    """Small running z-score normalizer mirroring VecNormalize-style updates."""

    def __init__(self, feature_dim, clip_obs=10.0, epsilon=1e-8):
        feature_dim = int(feature_dim)
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        self.feature_dim = feature_dim
        self.clip_obs = float(clip_obs)
        self.epsilon = float(epsilon)
        self.count = 0.0
        self.mean = np.zeros(feature_dim, dtype=float)
        self.m2 = np.zeros(feature_dim, dtype=float)

    @property
    def var(self):
        return self.m2 / max(self.count, 1.0)

    def update(self, values):
        values = np.asarray(values, float).reshape(-1)
        if values.size != self.feature_dim:
            raise ValueError("RunningFeatureNormalizer input dimension mismatch.")
        self.count += 1.0
        delta = values - self.mean
        self.mean += delta / self.count
        delta2 = values - self.mean
        self.m2 += delta * delta2

    def normalize(self, values, update=True):
        values = np.asarray(values, float).reshape(-1)
        if values.size != self.feature_dim:
            raise ValueError("RunningFeatureNormalizer input dimension mismatch.")
        if update:
            self.update(values)
        normed = (values - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normed, -self.clip_obs, self.clip_obs).astype(np.float32)

    def export_state(self):
        return {
            "count": float(self.count),
            "mean": self.mean.astype(np.float32).copy(),
            "var": self.var.astype(np.float32).copy(),
            "clip_obs": float(self.clip_obs),
            "epsilon": float(self.epsilon),
        }


class PhysicalXhatStateConditioner:
    def __init__(self, mode="fixed_minmax", clip_obs=10.0, epsilon=1e-8):
        mode = str(mode).strip().lower()
        if mode not in VALID_BASE_STATE_NORM_MODES:
            raise ValueError(f"Unsupported base_state_norm_mode: {mode}")
        self.mode = mode
        self.clip_obs = float(clip_obs)
        self.epsilon = float(epsilon)
        self._normalizer = None

    def transform(self, base_state, x_d_states, n_outputs, update=True):
        base_state = np.asarray(base_state, float).reshape(-1).copy()
        x_d_states = np.asarray(x_d_states, float).reshape(-1)
        n_outputs = int(n_outputs)
        n_phys = int(x_d_states.size - n_outputs)
        if n_phys < 0:
            raise ValueError("n_outputs cannot exceed the augmented state dimension.")
        if self.mode == "fixed_minmax" or n_phys == 0:
            return base_state.astype(np.float32), self.export_state()

        if self._normalizer is None or self._normalizer.feature_dim != n_phys:
            self._normalizer = RunningFeatureNormalizer(
                feature_dim=n_phys,
                clip_obs=self.clip_obs,
                epsilon=self.epsilon,
            )
        base_state[:n_phys] = self._normalizer.normalize(x_d_states[:n_phys], update=update)
        return base_state.astype(np.float32), self.export_state()

    def export_state(self):
        payload = {
            "mode": self.mode,
            "clip_obs": float(self.clip_obs),
            "epsilon": float(self.epsilon),
        }
        if self._normalizer is not None:
            payload.update(self._normalizer.export_state())
        return payload


def create_state_conditioner(mode="fixed_minmax", clip_obs=10.0, epsilon=1e-8):
    return PhysicalXhatStateConditioner(mode=mode, clip_obs=clip_obs, epsilon=epsilon)


def transform_mismatch_feature(values, *, mode="hard_clip", mismatch_clip=3.0, tanh_scale=3.0, post_clip=None):
    mode = str(mode).strip().lower()
    if mode not in VALID_MISMATCH_TRANSFORM_MODES:
        raise ValueError(f"Unsupported mismatch_feature_transform_mode: {mode}")
    values = np.asarray(values, float)

    if mode == "hard_clip":
        transformed = values.copy()
        if mismatch_clip is not None:
            clip_val = float(mismatch_clip)
            transformed = np.clip(transformed, -clip_val, clip_val)
    elif mode == "soft_tanh":
        scale = float(tanh_scale)
        if scale <= 0.0:
            raise ValueError("mismatch_transform_tanh_scale must be positive.")
        transformed = scale * np.tanh(values / scale)
    else:
        transformed = np.sign(values) * np.log1p(np.abs(values))

    if post_clip is not None:
        clip_val = float(post_clip)
        transformed = np.clip(transformed, -clip_val, clip_val)
    return transformed.astype(np.float32)


def normalize_observer_update_alignment(mode):
    mode = str(mode or "legacy_previous_measurement").strip().lower()
    if mode == "predictor_corrector_current":
        mode = "current_measurement_corrector"
    if mode not in VALID_OBSERVER_ALIGNMENT_MODES:
        raise ValueError(
            "observer_update_alignment must be 'legacy_previous_measurement' or 'current_measurement_corrector'."
        )
    return mode


def update_observer_state(
    *,
    A,
    B,
    C,
    L,
    x_prev,
    u_dev,
    y_prev_scaled,
    y_current_scaled,
    observer_update_alignment,
):
    alignment = normalize_observer_update_alignment(observer_update_alignment)
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)
    L = np.asarray(L, float)
    x_prev = np.asarray(x_prev, float).reshape(-1)
    u_dev = np.asarray(u_dev, float).reshape(-1)
    y_prev_scaled = np.asarray(y_prev_scaled, float).reshape(-1)
    y_current_scaled = np.asarray(y_current_scaled, float).reshape(-1)

    yhat_current = C @ x_prev
    if alignment == "current_measurement_corrector":
        x_pred = A @ x_prev + B @ u_dev
        y_pred_next = C @ x_pred
        x_next = x_pred + L @ (y_current_scaled - y_pred_next).T
    else:
        x_next = A @ x_prev + B @ u_dev + L @ (y_prev_scaled - yhat_current).T

    return x_next.astype(np.float32), yhat_current.astype(np.float32), alignment
