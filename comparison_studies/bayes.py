"""Two-parameter Gaussian-process expected-improvement MPC calibration."""
from __future__ import annotations
import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

LOW, HIGH = .75, 2.
INITIAL = [(1.,1.), (LOW,LOW), (LOW,HIGH), (HIGH,LOW), (HIGH,HIGH)]


def tune(evaluate, *, seed=7, trials=40, callback=None):
    if trials < 5:
        raise ValueError("Calibration needs all five initial settings.")
    sampler = qmc.Sobol(2, scramble=True, seed=seed)
    records, X, costs = [], [], []
    log_lo, log_range = np.log(LOW), np.log(HIGH/LOW)
    nominal_cost = None
    for trial in range(trials):
        if trial < 5:
            parameters = np.asarray(INITIAL[trial])
        else:
            kernel = ConstantKernel(1., (1e-3,1e3)) * Matern([1.,1.], (1e-2,10.), nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True,
                                          n_restarts_optimizer=2, random_state=seed)
            gp.fit(np.asarray(X), np.asarray(costs))
            candidates = sampler.random(4096)
            novel = np.min(np.max(np.abs(candidates[:,None,:]-np.asarray(X)[None,:,:]), axis=2), axis=1) > 1e-6
            candidates = candidates[novel]
            mean, std = gp.predict(candidates, return_std=True)
            improvement = min(costs)-mean-.01*max(float(np.std(costs)),1e-12)
            z = improvement/np.maximum(std,1e-12)
            ei = improvement*ndtr(z) + std*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
            parameters = np.exp(log_lo + log_range*candidates[int(np.argmax(ei))])
        cost, details = evaluate(float(parameters[0]), float(parameters[1]))
        valid = bool(np.isfinite(cost) and details.get("valid", True))
        if trial == 0:
            if not valid:
                raise RuntimeError("Nominal BO calibration trial failed; cannot define comparison.")
            nominal_cost = float(cost)
        observed_cost = float(cost) if valid else nominal_cost + 100*max(1.,abs(nominal_cost))
        record = {"trial":trial+1, "alpha":float(parameters[0]), "beta":float(parameters[1]),
                  "cost":observed_cost, "valid":valid, "details":details}
        records.append(record)
        X.append((np.log(parameters)-log_lo)/log_range)
        costs.append(observed_cost)
        if callback is not None:
            callback(records)
    best = min((r for r in records if r["valid"]), key=lambda r:r["cost"])
    return best, records
