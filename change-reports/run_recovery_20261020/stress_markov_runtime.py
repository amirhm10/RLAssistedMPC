"""Bounded real Markov LS and BLAS checks, without opening Aspen."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import review_blas_runtime as runtime


def fixture():
    import numpy as np
    import utils.markov_runner as markov
    rng = np.random.default_rng(23)
    A = np.diag([.75, .8, .85, .9])
    B = rng.normal(size=(4, 2)) * .1
    C = rng.normal(size=(2, 4))
    predict_h, control_horizon = 6, 3
    blocks = np.array([C @ np.linalg.matrix_power(A, lag) @ B for lag in range(predict_h)])
    basis = np.zeros((4, *blocks.shape))
    for k in range(4):
        basis[k, :, k // 2, k % 2] = blocks[:, k // 2, k % 2]
    return dict(z0=np.zeros(4), bounds=[(-.5, .5)] * 4,
                history={"u_dev_log": rng.normal(size=(64, 2)),
                         "y_scaled_dev": rng.normal(size=(64, 2)),
                         "xhat_before": rng.normal(size=(64, 4))},
                m_blocks=blocks, basis_blocks=basis,
                G0=markov.build_toeplitz_from_markov(blocks, predict_h, control_horizon),
                A=A, C=C, predict_h=predict_h, control_horizon=control_horizon,
                Wy=np.diag(np.linspace(.5, 2., predict_h * 2)), lambda_z=.1,
                current_step=50, prediction_window=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--seconds", type=float, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.private:
        runtime.activate()
    else:
        from run_polymer_e3_recovery import environment
        import os
        os.environ.update(environment())
    import numpy as np
    import torch
    import utils.markov_runner as markov
    import scipy.optimize as optimize
    torch.set_num_threads(1)
    info = runtime.runtime_info() if args.private else {"numpy": np.__version__}
    params = fixture()
    first = markov.fit_markov_ls_correction(**params)
    initial_score = markov.prediction_improvement_score(z=params["z0"], **{
        k: v for k, v in params.items() if k not in ("z0", "bounds")})
    assert first[1].success and np.isfinite(first[0]).all()
    assert first[2]["corrected_sse"] + .1 * np.sum(first[0] ** 2) <= initial_score["corrected_sse"] + 1e-8
    rng = np.random.default_rng(23023)
    began = time.perf_counter()
    linear = fits = 0
    max_residual = max_matvec_error = 0.
    while time.perf_counter() - began < args.seconds:
        for size in (4, 12, 24):
            raw = rng.normal(size=(size, size))
            matrix = raw.T @ raw + np.eye(size)
            vector = rng.normal(size=size)
            expected = np.einsum("ij,j->i", matrix, vector, optimize=False)
            max_matvec_error = max(max_matvec_error, float(np.max(np.abs(matrix @ vector - expected))))
            solved = np.linalg.solve(matrix, vector)
            max_residual = max(max_residual, float(np.max(np.abs(matrix @ solved - vector))))
            # Noncontiguous/Fortran layouts exercise GEMV as in the fatal LS stack.
            for view in (matrix.T, np.asfortranarray(matrix), matrix[::-1, ::-1]):
                assert np.allclose(view @ vector, np.einsum("ij,j->i", view, vector, optimize=False), rtol=1e-12, atol=1e-12)
            linear += 1
        z, result, score = markov.fit_markov_ls_correction(**params)
        assert result.success and np.isfinite(z).all() and np.isfinite(score["score"])
        assert np.allclose(z, first[0], rtol=1e-10, atol=1e-10)
        # Retain the MPC optimizer and PyTorch coexistence check too.
        result = optimize.minimize(lambda x: float(np.sum((x - .2) ** 2)), np.zeros(4),
                                   method="SLSQP", bounds=[(-1., 1.)] * 4)
        assert result.success and np.max(np.abs(result.x - .2)) < 1e-6
        tensor = torch.tensor(matrix, dtype=torch.float64, requires_grad=True)
        (tensor @ tensor.T).sum().backward()
        assert torch.isfinite(tensor.grad).all()
        fits += 1
    assert max_residual < 1e-9 and max_matvec_error < 1e-9
    record = {"state": "passed", "private": args.private, "seconds": time.perf_counter() - began,
              "markov_LBFGSB_fits": fits, "linear_batches": linear, "max_linear_residual": max_residual,
              "max_matvec_error": max_matvec_error, "reference_z": first[0].tolist(),
              "reference_score": first[2], "runtime": info,
              "limitation": "Synthetic history through actual production LS implementation; no Aspen, no long-run stability proof."}
    record["reference_score"] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in record["reference_score"].items()}
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k not in ("runtime", "reference_score")}, indent=2))


if __name__ == "__main__":
    main()
