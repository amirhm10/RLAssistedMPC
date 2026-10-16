"""Diagnose the preserved ARX result or run a separately versioned repair.

Default: print the plan. --diagnose is an offline saved-data audit; --smoke is
a short technical check. --run starts a complete new 200-cycle ARX trajectory.
Nothing edits or overwrites the historical result or original controller.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parent
HISTORICAL = ROOT/"Polymer/Results/comparisons/scenario1/20261014_033927_fd26b6/arx/training.pkl"


def source_hashes():
    from comparison_studies.common import source_hashes as previous
    result = previous()
    for path in (ROOT/"revision_comparisons/arx.py", ROOT/"revision_comparisons/runtime.py", Path(__file__).resolve()):
        result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def diagnose(path, output, replay=False):
    import numpy as np
    from comparison_studies.common import write_json
    from comparison_studies.arx import LAGS, IncrementalARX
    from revision_comparisons.arx import RegularizedARX, dc_gains
    path, output = Path(path), Path(output)
    with path.open("rb") as stream:
        data = pickle.load(stream)
    theta = np.asarray(data["comparison_arx_theta"], float)
    y, u = np.asarray(data["y"], float), np.asarray(data["u"], float)
    size = int(data["time_in_sub_episodes"])
    initial = data["comparison_arx_initial_theta"]
    nominal_gains = np.array([dc_gains(t, i) for i, t in enumerate(initial)])
    gains = []
    radii = []
    start = 0
    for i in range(2):
        width = len(initial[i])
        row = theta[:, start:start+width]
        denom = 1-row[:, 0]-row[:, 1]
        gains.append(np.stack([np.sum(row[:, [2+j for j, (channel, _) in enumerate(LAGS[i]) if channel == c]], axis=1)
                               / denom for c in range(2)], axis=1))
        disc = np.emath.sqrt(row[:, 0]**2+4*row[:, 1])
        radii.append(np.maximum(np.abs((row[:, 0]+disc)/2), np.abs((row[:, 0]-disc)/2)))
        start += width
    gains = np.stack(gains, axis=1)
    reversed_temperature_monomer = gains[:, 1, 1]*nominal_gains[1, 1] <= 0
    first = np.flatnonzero(reversed_temperature_monomer)
    diagnostics = {
        "source": str(path.resolve()), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "steps": len(theta), "nominal_dc_gains_scaled": nominal_gains,
        "all_admitted_poles_inside_unit_circle": bool(np.all(np.asarray(radii) < 1)),
        "temperature_monomer_reversed_gain_steps": int(reversed_temperature_monomer.sum()),
        "first_reversed_gain_step_zero_based": int(first[0]) if len(first) else None,
        "first_reversed_gain_cycle": int(first[0]//size+1) if len(first) else None,
        "last_cycle_temperature_min_max_K": [float(y[-size:, 1].min()), float(y[-size:, 1].max())],
        "last_cycle_mean_reward": float(np.mean(data["rewards_step"][-size:])),
        "final_dc_gains_scaled": gains[-1],
        "interpretation": "Pole stability admitted predictors with reversed manipulated-input gains. This establishes an inadequate model-admission rule; closed-loop data alone do not isolate every cause of parameter drift.",
        "repair": "Persistent nominal ridge prior plus gain direction/magnitude admission; last valid predictor retained.",
        "production_repair_run_completed": False,
        "cycles": []}
    for cycle in range(1, len(theta)//size+1):
        sl = slice((cycle-1)*size, cycle*size)
        diagnostics["cycles"].append({"cycle": cycle, "mean_reward": float(np.mean(data["rewards_step"][sl])),
            "temperature_min_max_K": [float(y[sl, 1].min()), float(y[sl, 1].max())],
            "reversed_temperature_monomer_steps": int(reversed_temperature_monomer[sl].sum()),
            "final_dc_gains_scaled": gains[cycle*size-1]})
    if replay:
        lo, hi = np.asarray(data["data_min"], float), np.asarray(data["data_max"], float)
        yss = (np.asarray(data["steady_states"]["y_ss"])-lo[2:])/(hi[2:]-lo[2:])
        uss = (np.asarray(data["steady_states"]["ss_inputs"])-lo[:2])/(hi[:2]-lo[:2])
        ys = (y-lo[2:])/(hi[2:]-lo[2:])-yss
        us = (u-lo[:2])/(hi[:2]-lo[:2])-uss
        model = RegularizedARX(initial, ys[0], np.zeros(2))
        old_model = IncrementalARX(initial, ys[0], np.zeros(2))
        largest_covariance_diagonal = np.zeros(2)
        historical_coefficient_max_difference = 0.
        error_sums = np.zeros(2)
        began = time.perf_counter()
        for k in range(len(us)):
            predicted = model.predict(us[k:k+1], 1)[0]
            error_sums += (predicted-ys[k+1])**2
            model.observe(ys[k+1], us[k])
            old_model.observe(ys[k+1], us[k])
            largest_covariance_diagonal = np.maximum(largest_covariance_diagonal,
                [float(np.max(np.diag(p))) for p in old_model.covariance])
            historical_coefficient_max_difference = max(historical_coefficient_max_difference,
                float(np.max(np.abs(np.concatenate(old_model.theta)-theta[k]))))
            if (k+1) % 20000 == 0:
                print(f"Offline ARX replay: {k+1}/{len(us)} observations", flush=True)
        diagnostics["repair_offline_replay"] = dict(model.diagnostics(), steps=len(us),
            observed_input_one_step_rmse_scaled=np.sqrt(error_sums/len(us)),
            seconds=time.perf_counter()-began,
            limitation="Historical failed-controller data, not a new closed-loop performance result.")
        diagnostics["historical_algorithm_replay"] = {
            "max_covariance_diagonal_by_output": largest_covariance_diagonal,
            "configured_covariance_eigenvalue_cap": 1e8,
            "max_abs_coefficient_difference_from_saved_trace": historical_coefficient_max_difference,
            "scope": "Same recorded inputs/outputs replayed through the original RLS equations in the current numerical runtime"}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"arx_diagnosis.json", diagnostics)
    print(json.dumps({k: v for k, v in diagnostics.items() if k not in {"cycles", "repair_offline_replay"}},
                     default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x), indent=2))
    return 0


def worker(path):
    import numpy as np
    import torch
    from comparison_studies.common import prepare_nominal, episode_bundle, solver_audit, write_json
    from comparison_studies.native import arx_run
    from comparison_studies.analysis import metrics
    from revision_comparisons.arx import RegularizedARX
    path = Path(path)
    job, output = json.loads(path.read_text(encoding="utf-8")), path.parent
    status = {"state": "running", "pid": os.getpid()}
    write_json(output/"status.json", status)
    try:
        if source_hashes() != job["source_hashes"]:
            raise RuntimeError("ARX revision dependencies changed before execution.")
        from revision_comparisons.runtime import seed_all, metadata
        seed_all(job["seed"])
        torch.set_num_threads(1)
        write_json(output/"runtime_metadata.json", metadata())
        namespace = prepare_nominal(job["seed"], output)
        bundle = episode_bundle(namespace, job["smoke"])
        predictors = []
        def factory(*args, **kwargs):
            model = RegularizedARX(*args, **kwargs)
            predictors.append(model)
            return model
        began = time.perf_counter()
        with patch("comparison_studies.arx.IncrementalARX", factory), solver_audit() as audit:
            result = arx_run(namespace, bundle)
        result.update(comparison_method="arx_regularized_v1", comparison_seed=job["seed"],
                      comparison_smoke=job["smoke"], comparison_source_hashes=job["source_hashes"],
                      comparison_arx_revision=predictors[0].diagnostics(), comparison_solver_audit=np.asarray(audit),
                      comparison_worker_seconds=time.perf_counter()-began)
        with (output/"training.pkl").open("xb") as stream:
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(output/"arx_configuration.json", result["comparison_arx_revision"])
        write_json(output/"resolved_defaults.json", namespace["NB"])
        if not job["smoke"]:
            write_json(output/"metrics.json", metrics(result))
        if source_hashes() != job["source_hashes"]:
            raise RuntimeError("ARX revision dependencies changed; raw trajectory retained.")
        status.update(state="completed", steps=int(result["nFE"]), smoke=job["smoke"],
                      trajectory_sha256=hashlib.sha256((output/"training.pkl").read_bytes()).hexdigest(),
                      solver_failed_calls=sum(not row[1] for row in audit))
        write_json(output/"status.json", status)
        return 0
    except BaseException as error:
        status.update(state="failed", error=str(error), error_type=type(error).__name__)
        write_json(output/"status.json", status)
        traceback.print_exc()
        return 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--replay", action="store_true", help="Replay saved observations offline through the repair.")
    p.add_argument("--source", type=Path, default=HISTORICAL)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.worker_job:
        return worker(args.worker_job)
    if not 0 <= args.seed <= 2**32-1:
        p.error("Seed must be an unsigned 32-bit integer.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6]
    output = (args.output_dir or ROOT/"Polymer/Results/comparisons/arx_revision"/stamp).resolve()
    if args.diagnose:
        return diagnose(args.source, output, args.replay)
    if not args.run and not args.smoke:
        print(json.dumps({"method": "arx_regularized_v1", "seed": args.seed,
                          "profile": "legacy_gradual_200", "cycles": 200, "production_steps": 160000,
                          "original_controllers_changed": False, "new_runs_started": False,
                          "repair": "persistent nominal RLS prior; nominal gain signs; gain ratios [.25,4]",
                          "instruction": "--diagnose --replay for saved-data audit; --smoke for technical check; --run for production"}, indent=2))
        return 0
    from comparison_studies.common import write_json
    from revision_comparisons.runtime import numerical_environment
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"job.json", {"method": "arx_regularized_v1", "seed": args.seed,
                                 "smoke": args.smoke, "source_hashes": source_hashes()})
    with (output/"worker.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                                 "--worker-job", str(output/"job.json")], cwd=ROOT,
                                env=numerical_environment(1), stdout=log, stderr=subprocess.STDOUT)
        print(f"ARX revision PID={proc.pid}; log: {output/'worker.log'}", flush=True)
        with (output/"worker.log").open(encoding="utf-8", errors="replace") as reader:
            while proc.poll() is None:
                chunk = reader.read()
                if chunk:
                    print(chunk, end="", flush=True)
                time.sleep(1.)
            print(reader.read(), end="", flush=True)
        return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
