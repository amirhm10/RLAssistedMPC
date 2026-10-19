"""Bounded numerical-runtime diagnostic; no simulator, training or source edits."""
from pathlib import Path
import json
import time

import numpy as np
from numpy._core import _multiarray_umath
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize
import torch
from threadpoolctl import threadpool_info


def main():
    torch.set_num_threads(1)
    pools = threadpool_info()
    blas = [p for p in pools if p["internal_api"] == "openblas"]
    assert blas and all(p["architecture"] == "Nehalem" and p["num_threads"] == 1 for p in blas)
    assert _multiarray_umath.__cpu_features__["X86_V3"] is False
    rng = np.random.default_rng(20261018)
    began = time.perf_counter()
    iterations = optimizer_calls = 0
    max_linear_residual = max_optimizer_error = 0.
    while time.perf_counter() - began < 40:
        size = (4, 6, 12, 24)[iterations % 4]
        raw = rng.standard_normal((size, size))
        matrix = raw.T @ raw + np.eye(size)
        target = rng.standard_normal(size)
        solved = np.linalg.solve(matrix, target)
        residual = float(np.max(np.abs(matrix @ solved - target)))
        max_linear_residual = max(max_linear_residual, residual)
        assert residual < 1e-9
        inverse = np.linalg.inv(matrix)
        assert np.max(np.abs(matrix @ inverse - np.eye(size))) < 1e-9
        if iterations % 10 == 0:
            desired = rng.uniform(-.5, .5, 6)
            result = minimize(lambda x: float(np.dot(x - desired, x - desired)), np.zeros(6),
                              method="SLSQP", bounds=[(-1., 1.)] * 6,
                              options={"maxiter": 80, "ftol": 1e-10})
            assert result.success
            max_optimizer_error = max(max_optimizer_error, float(np.max(np.abs(result.x - desired))))
            assert max_optimizer_error < 1e-5
            stable = np.eye(6) * .8
            covariance = solve_discrete_lyapunov(stable, np.eye(6))
            assert np.all(np.isfinite(covariance))
            tensor = torch.tensor(matrix, dtype=torch.float64, requires_grad=True)
            (tensor @ tensor.T).sum().backward()
            assert torch.isfinite(tensor.grad).all()
            optimizer_calls += 1
        iterations += 1
    record = {"state": "passed", "duration_seconds": time.perf_counter() - began,
              "numpy_solve_inverse_pairs": iterations, "SLSQP_Lyapunov_PyTorch_batches": optimizer_calls,
              "max_linear_residual": max_linear_residual, "max_optimizer_error": max_optimizer_error,
              "threadpools": pools, "numpy_x86_v3_enabled": False,
              "limitation": "Bounded CPU/import/BLAS/optimizer stress; does not prove multi-day training or Aspen stability."}
    path = Path(__file__).with_name("runtime_stress.json")
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
