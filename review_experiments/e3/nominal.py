"""Independent nominal OF-MPC data shared by both controller configurations."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np

from .common import PREDICTORS, json_safe, sha256, write_json
from .prediction import forecasts, observer_history
from .runtime import instrument_function


@contextmanager
def monitor_nominal(stop_file=None):
    from utils import mpc_baseline_runner as module
    original = module.run_offsetfree_mpc
    diagnostics = []

    def before(env):
        if stop_file is not None and Path(stop_file).exists():
            raise KeyboardInterrupt("E3 parent requested a graceful stop.")

    def after(env):
        diagnostics.append({"step": int(env["i"]), "solver_success": bool(env["sol"].success),
                            "solver_status": int(env["sol"].status)})

    module.run_offsetfree_mpc = instrument_function(original, before, after)
    try:
        yield diagnostics
    finally:
        module.run_offsetfree_mpc = original


def save_nominal(namespace, output, solver_diagnostics, *, smoke=False):
    output = Path(output)
    bundle = namespace["result_bundle"]
    if bundle["run_mode"] != "nominal" or bundle.get("disturbance_profile") is not None:
        raise ValueError("E3 nominal probe must use a fresh undisturbed plant.")
    y, u = np.asarray(bundle["y"]), np.asarray(bundle["u"])
    if y.shape[0] != len(u) + 1 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(u)):
        raise ValueError("Nominal probe returned nonfinite/misaligned physical data.")
    cycle = int(bundle["time_in_sub_episodes"])
    if len(u) != 2 * cycle:
        raise ValueError("Nominal probe must contain exactly two reference cycles.")
    path = output / "nominal.npz"
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, y=y, u=u, reference_scaled=bundle["y_sp"],
                        cycle_steps=cycle, delta_t=bundle["delta_t"],
                        **{key: np.asarray(namespace[key]) for key in ("A_aug", "B_aug", "C_aug", "data_min", "data_max")})
    metadata = {"state": "completed", "sha256": sha256(path), "run_mode": "nominal",
                "cycles": 2, "initialization_cycles": 1, "cycle_steps": cycle,
                "delta_t_hours": float(bundle["delta_t"]), "smoke": smoke,
                "solver_failures": sum(not row["solver_success"] for row in solver_diagnostics),
                "solver_diagnostics": solver_diagnostics,
                "controller": "unchanged nominal OF-MPC; no RL; no imposed disturbances",
                "system_setup": namespace["NB"]["system_setup"],
                "baseline_config": namespace["mpc_cfg"]}
    write_json(output / "nominal_metadata.json", metadata)
    return path


def nominal_origins(cycle, horizon, samples=50):
    # Split evenly over both reference holds, keeping every future in its hold.
    if cycle % 2 or samples < 2:
        raise ValueError("Nominal validation needs two equal reference holds and at least two origins.")
    origins = []
    for index in range(2):
        start, end = cycle + index * (cycle // 2), cycle + (index + 1) * (cycle // 2)
        pool = np.arange(start, end - horizon + 1, horizon, dtype=int)
        count = samples // 2 + int(index < samples % 2)
        if len(pool) < count:
            raise ValueError(f"Nominal validation budget exceeds complete non-overlapping futures: {len(pool)} < {count}.")
        origins.extend(pool[np.linspace(0, len(pool) - 1, count, dtype=int)].tolist())
    return origins


def evaluate_nominal(capture_dir, nominal_file, output, horizons, samples=50):
    """Evaluate two preselected frozen correction sets, with fresh causal states."""
    capture_dir, nominal_file, output = Path(capture_dir), Path(nominal_file), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    status = json.loads((capture_dir / "capture_status.json").read_text())
    if status["state"] != "completed":
        raise ValueError("Nominal checkpoint evaluation requires complete training capture.")
    probe_meta = json.loads(nominal_file.with_name("nominal_metadata.json").read_text())
    dataset_hash = sha256(nominal_file)
    if probe_meta["sha256"] != dataset_hash or probe_meta["state"] != "completed":
        raise ValueError("Nominal dataset hash/status mismatch.")
    with np.load(nominal_file, allow_pickle=False) as probe:
        data = {key: probe[key] for key in probe.files}
    origins = nominal_origins(int(data["cycle_steps"]), max(horizons), samples)
    training = [json.loads(line) for line in (capture_dir / "records.jsonl").read_text().splitlines()]
    records = []
    for checkpoint, trace in status["checkpoints"].items():
        source = next(row for row in training if row["trace"] == trace)
        with np.load(capture_dir / trace, allow_pickle=False) as saved:
            model = {key: saved[key] for key in saved.files if not key.startswith("forecast_")}
        for key, probe_key in (("A", "A_aug"), ("B", "B_aug"), ("C", "C_aug"),
                               ("data_min", "data_min"), ("data_max", "data_max")):
            if model[key].shape != data[probe_key].shape or not np.allclose(model[key], data[probe_key], rtol=1e-10, atol=1e-12):
                raise ValueError(f"{key}: checkpoint and nominal probe use different identification assets/scaling.")
        states = observer_history(model, data["y"], data["u"])
        for origin in origins:
            u, y = data["u"][origin:origin + max(horizons)], data["y"][origin + 1:origin + max(horizons) + 1]
            predicted, failures = forecasts(model, states[origin], u)
            path = output / f"{checkpoint}_{origin:07d}.npz"
            np.savez_compressed(path, y_future=y, u_future=u, data_min=model["data_min"], data_max=model["data_max"],
                                **{f"forecast_{name}": predicted[name] for name in PREDICTORS})
            records.append({**{key: source[key] for key in ("plant", "category", "seed", "smoke", "executed_source_code", "gate_source_code")},
                            "dataset": "nominal_checkpoint", "window": checkpoint,
                            "checkpoint_step": source["step"], "step": origin,
                            "nominal_dataset_sha256": dataset_hash, "predictor_failures": failures,
                            "nonfinite_truth": False, "trace": path.name,
                            "solver_failures_in_nominal_dataset": probe_meta["solver_failures"]})
    with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
    write_json(output / "validation_metadata.json", {
        "nominal_dataset_sha256": dataset_hash, "origins": origins, "horizons": horizons,
        "scope": "frozen correction sets; no policy evaluation or correction refit; shared nominal free response",
        "initialization": "causal observer with captured update convention; first cycle excluded",
        "checkpoint_selection": "last sampled origin in mature pre-change and adapted windows; no outcome selection",
        "saved": len(records), "smoke": probe_meta["smoke"],
    })
    return records
