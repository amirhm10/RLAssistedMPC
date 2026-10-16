"""Nominal-regularized RLS with explicit admission of usable ARX models.

This module deliberately leaves the historical comparator unchanged. A stable
autoregressive polynomial alone does not validate the input/output gains used
by MPC. Bounds below are fixed relative to the nominal identification, before
any revised closed-loop outcome is observed.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

from comparison_studies.arx import IncrementalARX, LAGS


def dc_gains(theta, output):
    """Steady input/output gains of the incremental ARX transfer function."""
    theta = np.asarray(theta, float)
    denominator = 1.0 - theta[0] - theta[1]
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return np.full(2, np.nan)
    return np.array([sum(b for b, (j, _) in zip(theta[2:], LAGS[output]) if j == channel)
                     / denominator for channel in range(2)])


class RegularizedARX(IncrementalARX):
    """Exponentially weighted LS with a persistent nominal ridge prior.

    H <- lambda H + phi phi' + (1-lambda) H0
    b <- lambda b + phi dy + (1-lambda) H0 theta0

    Thus the nominal prior does not disappear during long, weakly excited
    closed-loop operation. Candidate models additionally retain nominal gain
    directions and gains between 0.25 and 4 times their nominal magnitude.
    Rejected candidates leave the last admitted controller predictor intact.
    These engineering restrictions do not guarantee closed-loop stability.
    """

    def __init__(self, theta, current_y, current_u, *, forgetting=.995,
                 adaptive=True, prior_covariance=1000., gain_bounds=(.25, 4.)):
        super().__init__(theta, current_y, current_u, forgetting=forgetting, adaptive=adaptive)
        if not np.isfinite(prior_covariance) or prior_covariance <= 0:
            raise ValueError("The nominal prior covariance must be positive and finite.")
        low, high = map(float, gain_bounds)
        if not 0 < low <= 1 <= high:
            raise ValueError("Gain bounds must bracket the nominal gain and preserve its direction.")
        self.gain_bounds = (low, high)
        self.nominal_theta = [t.copy() for t in self.theta]
        self.prior_information = [np.eye(len(t)) / prior_covariance for t in self.theta]
        self.information = [h.copy() for h in self.prior_information]
        # The inherited covariance recursion is replaced by information-form
        # regularized LS. Do not expose its now-stale initialization as P_t.
        del self.covariance
        self.rhs = [h @ t for h, t in zip(self.information, self.theta)]
        self.nominal_gains = [dc_gains(t, i) for i, t in enumerate(self.theta)]
        if not all(self.admission_reason(t, i) == "accepted" for i, t in enumerate(self.theta)):
            raise ValueError("The nominal ARX model does not satisfy its own admission rules.")
        self.rejections = Counter()

    def admission_reason(self, theta, output):
        if not self.valid(theta):
            return "nonfinite_or_unstable_poles"
        candidate = dc_gains(theta, output)
        nominal = self.nominal_gains[output]
        if not np.all(np.isfinite(candidate)):
            return "nonfinite_dc_gain"
        nonzero = np.abs(nominal) > 1e-10
        if np.any(candidate[nonzero] * nominal[nonzero] <= 0):
            return "reversed_gain_direction"
        ratios = candidate[nonzero] / nominal[nonzero]
        if np.any((ratios < self.gain_bounds[0]) | (ratios > self.gain_bounds[1])):
            return "dc_gain_outside_nominal_bounds"
        if np.any(np.abs(candidate[~nonzero]) > 1e-10):
            return "unsupported_new_dc_coupling"
        return "accepted"

    def observe(self, y_next, u_executed):
        y_next, u_executed = np.asarray(y_next, float), np.asarray(u_executed, float)
        if not np.all(np.isfinite(y_next)) or not np.all(np.isfinite(u_executed)):
            raise ValueError("ARX observations must be finite.")
        dy_next = y_next - self.current_y
        du_new = np.vstack([u_executed - self.current_u, self.du[:-1]])
        self.samples += 1
        accepted = [False, False]
        if self.adaptive and self.samples >= 5:
            for i in range(2):
                phi = np.r_[self.dy[:, i], [du_new[lag-1, j] for j, lag in LAGS[i]]]
                if np.dot(phi, phi) <= 1e-12:
                    self.rejections["insufficient_regressor_energy"] += 1
                    continue
                prior = self.prior_information[i]
                h = (self.forgetting * self.information[i] + np.outer(phi, phi)
                     + (1-self.forgetting) * prior)
                rhs = (self.forgetting * self.rhs[i] + phi * dy_next[i]
                       + (1-self.forgetting) * (prior @ self.nominal_theta[i]))
                h = (h + h.T) / 2
                candidate = np.linalg.solve(h, rhs)
                self.information[i], self.rhs[i], self.estimates[i] = h, rhs, candidate
                reason = self.admission_reason(candidate, i)
                if reason == "accepted":
                    self.theta[i] = candidate.copy()
                    accepted[i] = True
                else:
                    self.rejections[reason] += 1
        self.dy = np.vstack([dy_next, self.dy[0]])
        self.du = du_new
        self.current_y, self.current_u = y_next.copy(), u_executed.copy()
        self.update_log.append(accepted)
        self.coefficient_log.append(np.concatenate(self.theta))

    def diagnostics(self):
        return {"revision": "persistent_nominal_prior_and_dc_gain_admission_v1",
                "forgetting": self.forgetting, "gain_ratio_bounds": self.gain_bounds,
                "nominal_gains": self.nominal_gains,
                "rejections": dict(self.rejections),
                "information_min_eigenvalues": [float(np.linalg.eigvalsh(h).min()) for h in self.information],
                "claim": "bounded model admission; no formal closed-loop stability guarantee"}
