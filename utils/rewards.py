import numpy as np


def make_reward_fn_relative_QR(
    data_min,
    data_max,
    n_inputs,
    k_rel,
    band_floor_phys,
    Q_diag,
    R_diag,
    tau_frac=0.7,
    gamma_out=0.5,
    gamma_in=0.5,
    beta=5.0,
    gate="geom",
    lam_in=1.0,
    bonus_kind="exp",
    bonus_k=12.0,
    bonus_p=0.6,
    bonus_c=20.0,
    reward_scale=0.01,
):
    """
    Reward with relative tracking bands in physical output space.

    The returned reward function matches the notebook behavior, with an
    explicit final reward scale so each system can keep its legacy range.
    """

    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    dy = np.maximum(data_max[n_inputs:] - data_min[n_inputs:], 1e-12)

    k_rel = np.asarray(k_rel, float)
    band_floor_phys = np.asarray(band_floor_phys, float)
    Q_diag = np.asarray(Q_diag, float)
    R_diag = np.asarray(R_diag, float)
    band_floor_scaled = band_floor_phys / np.maximum(dy, 1e-12)

    def _sigmoid(x):
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _phi(z, kind=bonus_kind, k=bonus_k, p=bonus_p, c=bonus_c):
        z = np.clip(z, 0.0, 1.0)
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
        raise ValueError("unknown bonus kind")

    def reward_fn(e_scaled, du_scaled, y_sp_phys=None):
        e_scaled = np.asarray(e_scaled, float)
        du_scaled = np.asarray(du_scaled, float)

        if y_sp_phys is None:
            band_scaled = band_floor_scaled
        else:
            y_sp_phys_arr = np.asarray(y_sp_phys, float)
            band_phys = np.maximum(k_rel * np.abs(y_sp_phys_arr), band_floor_phys)
            band_scaled = band_phys / np.maximum(dy, 1e-12)

        tau_scaled = tau_frac * band_scaled
        abs_e = np.abs(e_scaled)
        s_i = _sigmoid((band_scaled - abs_e) / np.maximum(tau_scaled, 1e-12))

        if gate == "prod":
            w_in = float(np.prod(s_i, dtype=np.float64))
        elif gate == "mean":
            w_in = float(np.mean(s_i))
        elif gate == "geom":
            w_in = float(np.prod(s_i, dtype=np.float64) ** (1.0 / len(s_i)))
        else:
            raise ValueError("gate must be 'prod'|'mean'|'geom'")

        err_quad = np.sum(Q_diag * (e_scaled ** 2))
        err_eff = (1.0 - w_in) * err_quad + w_in * (lam_in * err_quad)
        move = np.sum(R_diag * (du_scaled ** 2))

        slope_at_edge = 2.0 * Q_diag * band_scaled
        overflow = np.maximum(abs_e - band_scaled, 0.0)
        inside_mag = np.minimum(abs_e, band_scaled)
        lin_out = (1.0 - w_in) * np.sum(gamma_out * slope_at_edge * overflow)
        lin_in = w_in * np.sum(gamma_in * slope_at_edge * inside_mag)

        qb2 = Q_diag * (band_scaled ** 2)
        z = abs_e / np.maximum(band_scaled, 1e-12)
        phi = _phi(z)
        bonus = w_in * beta * np.sum(qb2 * phi)

        return (-(err_eff + move + lin_out + lin_in) + bonus) * float(reward_scale)

    params = {
        "k_rel": k_rel,
        "band_floor_phys": band_floor_phys,
        "band_floor_scaled": band_floor_scaled,
        "Q_diag": Q_diag,
        "R_diag": R_diag,
        "tau_frac": tau_frac,
        "gamma_out": gamma_out,
        "gamma_in": gamma_in,
        "beta": beta,
        "gate": gate,
        "lam_in": lam_in,
        "bonus_kind": bonus_kind,
        "bonus_k": bonus_k,
        "bonus_p": bonus_p,
        "bonus_c": bonus_c,
        "reward_scale": float(reward_scale),
    }
    return params, reward_fn


def make_reward_fn_prototype_legacy(
    data_min,
    data_max,
    n_inputs,
    y_ss_scaled,
    Q_diag,
    R_diag,
    bonus_A=1000.0,
    bonus_B=1.0,
):
    """
    Prototype Markov reward used before the unified migration.

    This reproduces the older reward exactly in terms of:
    - scaled output error penalty
    - scaled input-move penalty
    - exponential bonus when every output's relative scaled error is within 5%

    The denominator for the percentage check is the scaled setpoint deviation,
    matching the prototype script behavior.
    """

    data_min = np.asarray(data_min, float)
    data_max = np.asarray(data_max, float)
    y_ss_scaled = np.asarray(y_ss_scaled, float).reshape(-1)
    q_diag = np.asarray(Q_diag, float).reshape(-1)
    r_diag = np.asarray(R_diag, float).reshape(-1)
    y_scale_min = data_min[int(n_inputs) :]
    y_scale_max = data_max[int(n_inputs) :]

    def reward_fn(e_scaled, du_scaled, y_sp_phys=None):
        e_scaled = np.asarray(e_scaled, float).reshape(-1)
        du_scaled = np.asarray(du_scaled, float).reshape(-1)

        if y_sp_phys is None:
            y_sp_scaled_dev = np.zeros_like(e_scaled)
        else:
            y_sp_phys_arr = np.asarray(y_sp_phys, float).reshape(-1)
            y_sp_scaled_dev = (y_sp_phys_arr - y_scale_min) / np.maximum(y_scale_max - y_scale_min, 1.0e-12)
            y_sp_scaled_dev = y_sp_scaled_dev - y_ss_scaled

        reward = -(np.sum(q_diag * (e_scaled**2)) + np.sum(r_diag * (du_scaled**2)))
        error_norm = np.abs(np.linalg.norm(e_scaled.reshape(1, -1), axis=0) / (y_sp_scaled_dev + 1.0e-15)) * 100.0
        if np.all(error_norm <= 5.0):
            reward += float(bonus_A) * np.exp(-float(bonus_B) * float(np.mean(error_norm)))
        return float(reward)

    params = {
        "mode": "prototype_legacy",
        "Q_diag": q_diag,
        "R_diag": r_diag,
        "bonus_A": float(bonus_A),
        "bonus_B": float(bonus_B),
        "uses_scaled_setpoint_denominator": True,
    }
    return params, reward_fn
