"""Exercise unchanged LS scoring/fitting under independently started runtimes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores", type=int, default=2000)
    parser.add_argument("--fits", type=int, default=100)
    args = parser.parse_args()
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info
    from utils.markov_runner import (compute_markov_blocks, make_markov_basis,
        build_toeplitz_from_markov, prediction_improvement_score, fit_markov_ls_correction)
    trace = sorted((ROOT / "Results/e3/20261015_042751_513bff/polymer_markov_s7/a1/capture/traces").glob("*.npz"))[0]
    with np.load(trace, allow_pickle=False) as source:
        A, B, C = (source[name].copy() for name in ("A", "B", "C"))
    rng = np.random.default_rng(71)
    n = 96
    u = rng.normal(0, .02, (n, B.shape[1]))
    states = np.zeros((n + 1, A.shape[0]))
    for k in range(n): states[k + 1] = A @ states[k] + B @ u[k]
    y = states @ C.T + rng.normal(0, .0005, (n + 1, C.shape[0]))
    fixtures = []
    for index, h in enumerate((4, 8, 12, 16, 20)):
        hc = min(h, 2 + index)
        blocks = compute_markov_blocks(A, B, C, h)
        basis, _ = make_markov_basis(blocks, "io_pair_gain")
        fixtures.append(dict(history={"u_dev_log": u, "y_scaled_dev": y, "xhat_before": states},
            m_blocks=blocks, basis_blocks=basis, G0=build_toeplitz_from_markov(blocks, h, hc),
            A=A, C=C, predict_h=h, control_horizon=hc, Wy=np.eye(h * C.shape[0]),
            lambda_z=.01, current_step=n - 1, prediction_window=5))
    began, scores, fits = time.perf_counter(), [], []
    for i in range(args.scores):
        value = prediction_improvement_score(z=np.full(4, .2 * np.sin(i)), **fixtures[i % len(fixtures)])
        if not np.isfinite(value["score"]): raise RuntimeError("Nonfinite score in numerical probe.")
        if i < 25: scores.append(value["score"])
    for i in range(args.fits):
        z, result, score = fit_markov_ls_correction(z0=np.zeros(4), bounds=[(-.35, .35)] * 4,
            **fixtures[i % len(fixtures)])
        if not np.all(np.isfinite(z)) or not np.isfinite(score["score"]):
            raise RuntimeError("Nonfinite fit in numerical probe.")
        if i < 10: fits.append({"z": z.tolist(), "score": score["score"], "success": bool(result.success)})
    report = {"numpy": np.__version__, "scipy": scipy.__version__, "threadpools": threadpool_info(),
        "score_calls": args.scores, "fit_calls": args.fits, "seconds": time.perf_counter() - began,
        "reference_trace": str(trace), "scores": scores, "fits": fits,
        "scope": "unchanged functions; captured plant matrices; synthetic finite measured history",
        "limitation": "Does not reconstruct the lost cycle-100 crash state or prove long-run stability."}
    with args.output.open("x", encoding="utf-8") as stream: json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ("score_calls", "fit_calls", "seconds")}), flush=True)


if __name__ == "__main__": main()
