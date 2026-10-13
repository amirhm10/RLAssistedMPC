"""Fixed-delay incremental ARX identification and history-based MPC prediction."""
from __future__ import annotations
from copy import deepcopy
import numpy as np

LAGS = (((0, 3), (0, 4), (0, 5), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 1), (1, 2)))


def nominal_coefficients(A, B, C):
    """Recover the nominal row recurrences, verifying the prescribed structure."""
    A, B, C = map(lambda x: np.asarray(x, float), (A, B, C))
    H = [np.zeros((2, 2))] + [C @ np.linalg.matrix_power(A, k - 1) @ B for k in range(1, 31)]
    result = []
    for i in range(2):
        design = np.array([[H[k-1][i,j], H[k-2][i,j]] for k in range(8, 30) for j in range(2)])
        target = np.array([H[k][i,j] for k in range(8, 30) for j in range(2)])
        aa = np.linalg.lstsq(design, target, rcond=None)[0]
        bb = np.array([H[lag][i,j] - aa[0] * H[lag-1][i,j]
                       - (aa[1] * H[lag-2][i,j] if lag >= 2 else 0) for j, lag in LAGS[i]])
        theta = np.r_[aa, bb]
        for k in range(1, 30):
            residual = H[k][i].copy() - aa[0] * H[k-1][i]
            if k >= 2:
                residual -= aa[1] * H[k-2][i]
            expected = np.zeros(2)
            for b, (j, lag) in zip(bb, LAGS[i]):
                if lag == k:
                    expected[j] += b
            if not np.allclose(residual, expected, rtol=1e-7, atol=1e-10):
                raise ValueError("Nominal predictor no longer matches the fixed ARX structure.")
        result.append(theta)
    return result


class IncrementalARX:
    def __init__(self, theta, current_y, current_u, *, forgetting=.995, adaptive=True):
        self.theta = [np.asarray(t, float).copy() for t in theta]
        self.estimates = deepcopy(self.theta)
        self.covariance = [np.eye(len(t)) * 1000 for t in theta]
        self.forgetting = float(forgetting)
        if not 0 < self.forgetting <= 1:
            raise ValueError("Forgetting must be in (0,1].")
        self.adaptive = bool(adaptive)
        self.current_y, self.current_u = np.asarray(current_y, float).copy(), np.asarray(current_u, float).copy()
        self.dy = np.zeros((2, 2))
        self.du = np.zeros((5, 2))
        self.samples = 0
        self.update_log = []
        self.coefficient_log = []

    @staticmethod
    def valid(theta):
        return bool(np.all(np.isfinite(theta)) and np.max(np.abs(np.roots([1., -theta[0], -theta[1]]))) < 1.)

    def observe(self, y_next, u_executed):
        y_next, u_executed = np.asarray(y_next, float), np.asarray(u_executed, float)
        dy_next = y_next - self.current_y
        du_new = np.vstack([u_executed - self.current_u, self.du[:-1]])
        self.samples += 1
        accepted = [False, False]
        if self.adaptive and self.samples >= 5:
            for i in range(2):
                phi = np.r_[self.dy[:, i], [du_new[lag-1,j] for j, lag in LAGS[i]]]
                if np.all(np.isfinite(phi)) and np.isfinite(dy_next[i]) and np.dot(phi, phi) > 1e-12:
                    P = self.covariance[i]
                    gain = P @ phi / (self.forgetting + phi @ P @ phi)
                    candidate = self.estimates[i] + gain * (dy_next[i] - phi @ self.estimates[i])
                    updated_P = (P - np.outer(gain, phi) @ P) / self.forgetting
                    vals, vectors = np.linalg.eigh((updated_P + updated_P.T) / 2)
                    self.covariance[i] = (vectors * np.clip(vals, 1e-8, 1e8)) @ vectors.T
                    if np.all(np.isfinite(candidate)):
                        self.estimates[i] = candidate
                    if self.valid(candidate):
                        self.theta[i] = candidate.copy()
                        accepted[i] = True
        self.dy = np.vstack([dy_next, self.dy[0]])
        self.du = du_new
        self.current_y, self.current_u = y_next.copy(), u_executed.copy()
        self.update_log.append(accepted)
        self.coefficient_log.append(np.concatenate(self.theta))

    def predict(self, U, horizon):
        """U is the native absolute scaled-deviation input sequence (Nc,2)."""
        U = np.asarray(U, float).reshape(-1, 2)
        dy, du, y, u = self.dy.copy(), self.du.copy(), self.current_y.copy(), self.current_u.copy()
        predictions = []
        for k in range(int(horizon)):
            proposed_u = U[min(k, len(U)-1)]
            du = np.vstack([proposed_u-u, du[:-1]])
            future_dy = np.array([self.theta[i] @ np.r_[dy[:,i], [du[lag-1,j] for j,lag in LAGS[i]]]
                                  for i in range(2)])
            y = y + future_dy
            predictions.append(y.copy())
            dy = np.vstack([future_dy, dy[0]])
            u = proposed_u
        return np.asarray(predictions)


class ARXMPC:
    """Use nominal matrices for diagnostic observer logs, ARX for optimization."""
    def __init__(self, nominal, predictor):
        self.nominal, self.predictor = nominal, predictor
        for key in ("A", "B", "C", "Q_out", "R_in", "NP", "NC"):
            setattr(self, key, getattr(nominal, key))

    def mpc_opt_fun(self, x, y_sp, u_prev_dev, x0_model):
        U = np.asarray(x[:2*self.NC], float).reshape(self.NC, 2)
        prediction = self.predictor.predict(U, self.NP)
        error = prediction - np.asarray(y_sp)
        delta_u = U - np.vstack([np.asarray(u_prev_dev), U[:-1]])
        return float(np.sum(error**2*self.Q_out) + np.sum(delta_u**2*self.R_in))
