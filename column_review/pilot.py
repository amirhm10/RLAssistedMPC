"""Native Aspen snapshot/repeated-continuation pilot, never a critic experiment.

The API is verified against the installed registered type library without live
connection. The actual restoration remains an explicit, future live pilot.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import pickle
import uuid

import numpy as np

from .common import digest, write


def profile(plant):
    """Known Stage(...).T/y API, spanning both controlled trays and the column."""
    trays = (1, 12, 24, 40, 60, 85)
    values = [float(plant.sim.Time), float(plant.feed.FmR.Value),
              float(plant.col.Reflux.FmR.Value), float(plant.col.QRebR.Value)]
    labels = ["simulation_time_h", "feed_flow", "native_reflux_flow", "native_reboiler_duty"]
    for tray in trays:
        values.extend((float(plant.col.Stage(tray).T.Value), float(plant.col.Stage(tray).y("C2H6").Value)))
        labels.extend((f"tray_{tray}_temperature", f"tray_{tray}_ethane_vapor_fraction"))
    array = np.asarray(values, float)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("Nonfinite internal-state snapshot diagnostic.")
    return labels, array


class NativeSnapshot:
    def __init__(self, plant, controller_state, output):
        self.plant = plant
        self.python_state = {"current_input": plant.current_input.copy(),
                             "current_output": plant.current_output.copy(),
                             "controller": deepcopy(controller_state)}
        self.labels, self.internal = profile(plant)
        self.name = "review_e1_" + uuid.uuid4().hex
        plant.sim.Results.TakeSnapshot(self.name)
        snapshot = plant.sim.Results.FindSnapshot(self.name)
        if isinstance(snapshot, (int, np.integer)):
            snapshot = plant.sim.Results.GetSnapshot(int(snapshot))
        self.native = snapshot
        if not bool(self.native.Converged):
            raise RuntimeError("Aspen reports a nonconverged native snapshot; it cannot validate E1 restoration.")
        self.path = Path(output) / (self.name + ".snp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.native.ExportasBinary(str(self.path))
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise RuntimeError("Aspen did not export the native snapshot to the requested private path.")

    def restore(self):
        self.plant.sim.Results.LoadSnapshot(self.native)
        self.plant.current_input = self.python_state["current_input"].copy()
        # Read native measurements; stale cached Python outputs cannot pass.
        self.plant.current_output = np.asarray([
            self.plant.col.Stage(24).y("C2H6").Value, self.plant.col.Stage(85).T.Value], float)
        _, actual = profile(self.plant)
        tolerance = np.array([1e-9, 1e-3, 1e-3, 1e-6] + [value for _ in range(6) for value in (1e-5, 1e-9)])
        if np.any(np.abs(actual - self.internal) > tolerance):
            raise RuntimeError("Native snapshot failed immediate time/feed/internal-profile restoration.")
        if not np.allclose(self.plant.current_output, self.python_state["current_output"], rtol=0, atol=[1e-9, 1e-5]):
            raise RuntimeError("Snapshot restored internal diagnostics but not the controlled outputs.")
        return deepcopy(self.python_state["controller"])


def paired_continuations(plant, snapshot, inputs, disturbances):
    runs = []
    for repeat in range(3):
        restored = snapshot.restore()
        outputs, internal = [], []
        for control, disturbance in zip(inputs, disturbances):
            plant.current_input = np.asarray(control, float).copy()
            previous_time = float(plant.sim.Time)
            plant.step(disturbances=np.atleast_1d(disturbance))
            if not np.isclose(float(plant.sim.Time) - previous_time, float(plant.delta_t), rtol=1e-6, atol=1e-8):
                raise RuntimeError("Aspen did not advance by the original controller sampling interval.")
            outputs.append(plant.current_output.copy())
            internal.append(profile(plant)[1])
        runs.append({"outputs": np.asarray(outputs), "internal": np.asarray(internal), "controller": restored})
    for run in runs[1:]:
        if not np.allclose(run["outputs"], runs[0]["outputs"], rtol=0, atol=[1e-8, 1e-4]):
            raise RuntimeError("Repeated identical snapshot continuations disagree in physical outputs.")
        tolerance = np.array([1e-8, 1e-3, 1e-3, 1e-6] + [value for _ in range(6) for value in (1e-4, 1e-8)])
        if np.any(np.abs(run["internal"] - runs[0]["internal"]) > tolerance):
            raise RuntimeError("Repeated continuations disagree in internal profiles/time/feed.")
        np.testing.assert_array_equal(run["controller"]["observer"], runs[0]["controller"]["observer"])
    return runs


def run(output, aspen, seed=7):
    from revision_comparisons.column import prepare, baseline, calibration_window
    from review_experiments.e3.runtime import aspen_resources
    from .typelib import inspect_api
    output = Path(output)
    write(output / "snapshot_api.json", inspect_api())
    with aspen_resources(aspen):
        namespace = prepare(seed, output, aspen)
        full = namespace["EPISODE_BUNDLE"]
        # A complete original cycle preserves baseline clock/observer semantics.
        prefix = baseline(namespace, calibration_window(full, 0, 400))
        if prefix.get("use_shifted_mpc_warm_start"):
            raise RuntimeError("This pilot requires extraction of the shifted MPC warm start before continuation.")
        plant = namespace["system"]
        controller = {"observer": prefix["xhatdhat"][:, -1].copy(),
                      "previous_input": plant.current_input.copy(),
                      "mpc_initial_guess": np.zeros(2 * int(namespace["markov_cfg"]["cont_h"]))}
        snapshot = NativeSnapshot(plant, controller, output / "preserved_snapshots")
        controls = np.repeat(plant.current_input[None], 24, axis=0)
        span = np.asarray(namespace["SYS"]["input_bounds"]["u_max"]) - namespace["SYS"]["input_bounds"]["u_min"]
        controls[8:16, 0] += .005 * span[0]
        controls[16:, 1] += .005 * span[1]
        controls = np.clip(controls, namespace["SYS"]["input_bounds"]["u_min"], namespace["SYS"]["input_bounds"]["u_max"])
        disturbances = full["disturbance_schedule"][400:424].copy()
        runs = paired_continuations(plant, snapshot, controls, disturbances)
        snapshot.restore()
        np.savez_compressed(output / "continuations.npz", inputs=controls, disturbances=disturbances,
                            **{f"repeat_{index}_{key}": row[key] for index, row in enumerate(runs)
                               for key in ("outputs", "internal")})
        with (output / "python_checkpoint.pkl").open("xb") as stream:
            pickle.dump(snapshot.python_state, stream, protocol=5)
        result = {"state": "completed", "pilot_passed": True, "prefix_steps": 400,
                  "continuation_steps_each": 24, "identical_continuations": 3,
                  "diagnostic_labels": snapshot.labels, "native_snapshot": str(snapshot.path.relative_to(output)),
                  "native_snapshot_sha256": digest(snapshot.path),
                  "scope": "Native plant-state restoration, measured/internal-profile repeatability and Python observer copy.",
                  "e1_critic_validation_performed": False, "e1_production_ready": False,
                  "remaining": "Integrate plant snapshots with all frozen agents, conditioning, gate histories and MPC state in an E1 adapter; select/approve final sampling budget."}
    # Successful context exit also verifies owned-case cleanup.
    write(output / "pilot_result.json", result)
    return result
